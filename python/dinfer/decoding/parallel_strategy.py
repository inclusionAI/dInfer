import math
import torch
import numpy as np
import torch.nn.functional as F

from .utils import add_gumbel_noise, get_num_transfer_tokens
import torch.distributed as dist


def broadcast_if_needed(x, src=0, group=None):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.broadcast(x, src=src)


@torch.no_grad()
@torch.compile(dynamic=True)
def get_transfer_index_hierarchy_fast_v2(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    mask_id,
    threshold=None,
    low_threshold=None,
):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float32), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(
                confidence[j], k=num_transfer_tokens[j]
            )
            transfer_index[j, select_index] = True

    else:
        for i in range(mask_index.shape[0]):

            mask_i = mask_index[i].int()
            conf_i = confidence[i]

            if low_threshold is not None:
                max_value, max_index = torch.max(conf_i, dim=0)
                if max_value < low_threshold:
                    transfer_index[i, max_index] = True
                    continue

            diff = torch.diff(
                torch.cat([mask_i[:1] * 0, mask_i, mask_i[-1:] * 0])
            )
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]

            if len(starts) > 0:
                max_indices = [
                    s + torch.argmax(conf_i[s:e])
                    for s, e in zip(starts.tolist(), ends.tolist())
                ]
                transfer_index[i, max_indices] = True

            if low_threshold is not None:
                transfer_index[i] = torch.logical_and(
                    transfer_index[i], conf_i > low_threshold
                )

        if threshold is not None:
            transfer_index = torch.logical_or(
                transfer_index, confidence > threshold
            )

    return x0, transfer_index


@torch.no_grad()
def get_transfer_index_hierarchy_remask(
    logits,
    temperature,
    mask_index,
    x,
    num_transfer_tokens,
    mask_id,
    threshold=None,
    low_threshold=None,
    remask_threshold=0.4,
):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
    )  # b, l

    lower_index = x0_p < remask_threshold
    remask_index = torch.logical_and(
        lower_index, torch.logical_not(mask_index)
    )
    mask_new = torch.logical_or(lower_index, mask_index)

    confidence = torch.where(mask_new, x0_p, float('-inf'))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    remask_cnt = remask_index.sum(dim=1)

    if num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(
                confidence[j], k=num_transfer_tokens[j]
            )
            transfer_index[j, select_index] = True

    else:
        for i in range(mask_new.shape[0]):
            mask_i = mask_new[i].int()
            conf_i = confidence[i]

            diff = torch.diff(
                torch.cat([mask_i[:1] * 0, mask_i, mask_i[-1:] * 0])
            )
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]

            if len(starts) > 0:
                max_indices = [
                    s + torch.argmax(conf_i[s:e])
                    for s, e in zip(starts.tolist(), ends.tolist())
                ]
                transfer_index[i, max_indices] = True

            if low_threshold is not None:
                transfer_index[i] = torch.logical_and(
                    transfer_index[i], conf_i > low_threshold
                )

            if threshold is not None:
                transfer_index[i] = torch.logical_or(
                    transfer_index[i], conf_i > threshold
                )

            gap = int((remask_cnt[i] + 1 - transfer_index[i].sum()).item())
            if gap > 0:
                conf_i[transfer_index[i]] = float('-inf')
                values, indices = torch.topk(
                    conf_i, gap, largest=True, sorted=False
                )
                transfer_index[i][indices] = True

    remask_index = torch.logical_and(
        remask_index, torch.logical_not(transfer_index)
    )
    x0[remask_index] = mask_id
    transfer_index[remask_index] = True

    return x0, transfer_index


def get_transfer_index_cache(
    logits,
    mask_index,
    x,
    block_end,
    num_transfer_tokens,
    temperature,
    remasking,
    threshold=None,
    minimal_topk=1,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits[mask_index].to(torch.float32), dim=-1).to(
            logits.dtype
        )
        x0_p = torch.squeeze(
            torch.gather(
                p,
                dim=-1,
                index=torch.unsqueeze(x0[mask_index], -1),
            ),
            -1,
        )  # b, l
        confidence = torch.full(
            x0.shape, -np.inf, device=x0.device, dtype=logits.dtype
        )
        confidence[mask_index] = x0_p
        confidence[:, block_end:] = -np.inf

    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        x0_p[:, block_end:] = -np.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
    else:
        raise NotImplementedError(remasking)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(minimal_topk, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    threshold=None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_dynamic(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    factor=1,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        ns = list(range(1, num_transfer_tokens[j] + 1))
        es = [factor / (n + 1) for n in ns]
        threshs = [1 - e for e in es]

        # at least one token is transferred
        threshs[0] = -1
        sorted_confidence = torch.sort(
            confidence[j][mask_index[j]], dim=-1, descending=True
        )[0]
        assert len(sorted_confidence) == len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i] < threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs) - 1:
            top_i += 1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index


class ParallelDecoder:
    """ This is a parallel decoder that decodes tokens in a block.
    """
    def __init__(
            self,
            temperature,
            remasking='low_confidence',
            mask_id=126336,
    ):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        pass

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.

        Parameters
        ----------
        logits : Tensor
            The logits in a block
        block_start : int
            The location of the starting token in the block
        block_end : int
            The location of the ending token in the block.
        x : Tensor
            The tensor where the decoded tokens are written to.
        """


# Parallel decoding only
@torch.compile(dynamic=True)
def get_transfer_index_threshold(
    logits,
    temperature,
    mask_index,
    x,
    mask_id,
    threshold,
    rm_mask=True,
    use_float64=False,
    **kwargs,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if use_float64:
        p = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        p = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l

    # gurantee the denoised token will not be the mask_id
    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    actual_threshold = (
        torch.max(confidence, dim=1)[0] - 1e-5
    ).clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold
    return x0, transfer_index


@torch.no_grad()
@torch.compile(dynamic=True)
def get_transfer_index_threshold_remask(
    logits,
    temperature,
    mask_index,
    x,
    mask_id,
    threshold,
    use_float64=False,
    fix_mask=None,
    **kwargs,
):
    """Similar to get_transfer_index_threshold but with remasking support.
    
    Tokens that are already decoded but have confidence below the given threshold
    will be remasked (set back to mask_id).
    Uses the given threshold directly (no adaptive threshold calculation).
    """
    # Keep the original mask positions for remasking decision.
    orig_mask_index = mask_index

    # 1) Sample token ids from gumbel-noised logits (sampling decision).
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0_sample = torch.argmax(logits_with_noise, dim=-1)  # b, l

    # 2) Compute confidence from *clean* logits (no gumbel):
    #    - masked positions: probability of the sampled token (x0_sample from gumbel sampling)
    #    - unmasked positions: probability of current token in x (used for remasking decision)
    if use_float64:
        p = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        p = F.softmax(logits.to(torch.float32), dim=-1)

    # Probability of the *sampled* token at each position:
    # - token id comes from gumbel sampling (x0_sample)
    # - probability is evaluated under clean logits (p)
    # Used for thresholding masked positions (deciding which masks to fill this step).
    x0_p = torch.squeeze(
        torch.gather(
            p, dim=-1, index=torch.unsqueeze(x0_sample.to(torch.long), -1)
        ),
        -1,
    )  # b, l

    # Probability of the *current* token already in the sequence (x) under clean logits (p).
    # Used for remasking: unmasked positions with low confidence will be set back to mask_id.
    x_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x.to(torch.long), -1)),
        -1,
    )  # b, l

    # Step 1: Default decode all mask positions
    x0 = torch.where(orig_mask_index, x0_sample, x)  # [B, L]

    # Step 2: Build unified confidence: mask positions use x0_p, non-mask positions use x_p
    unified_confidence = torch.where(orig_mask_index, x0_p, x_p)  # [B, L]
    
    # If fix_mask is provided, set confidence of fixed positions to +inf to prevent remasking
    if fix_mask is not None:
        # fix_mask should have shape [B, L] matching unified_confidence
        # Set fixed positions to +inf so they won't be selected for remasking (highest confidence)
        unified_confidence = torch.where(fix_mask, torch.full_like(unified_confidence, float('inf')), unified_confidence)

    # Step 3: Use the given threshold directly (no adaptive threshold)
    if isinstance(threshold, torch.Tensor):
        threshold_tensor = threshold.to(device=unified_confidence.device, dtype=unified_confidence.dtype)
        if threshold_tensor.dim() == 0:
            threshold_tensor = threshold_tensor.unsqueeze(-1)
    else:
        threshold_tensor = torch.tensor(threshold, device=unified_confidence.device, dtype=unified_confidence.dtype).unsqueeze(-1)

    # Step 4: Unified decision: confidence >= threshold means keep, < threshold means remask
    keep_mask = unified_confidence >= threshold_tensor  # [B, L]
    
    # Step 5: Remask positions with low confidence, but ensure mask count is strictly decreasing
    # Calculate how many masks we decoded this step
    decoded_cnt = orig_mask_index.sum(dim=1)  # [B] - all original masks were decoded
    
    # Candidates for remasking: positions with low confidence
    remask_candidates = ~keep_mask  # [B, L]
    cand_cnt = remask_candidates.sum(dim=1)  # [B]
    
    # Limit remask count: remask < decoded to ensure mask count strictly decreasing
    # This ensures: final_mask = orig_mask - decoded + remask < orig_mask
    max_remask = torch.minimum(decoded_cnt - 1, cand_cnt).clamp(min=0)  # [B]

    # Sort candidates by confidence (lowest first) and select top max_remask
    cand_conf = torch.where(remask_candidates, unified_confidence, torch.inf)  # Non-candidates set to +inf
    sorted_idx = torch.argsort(cand_conf, dim=1)  # [B, L]
    ranks = torch.argsort(sorted_idx, dim=1)      # ranks[pos] = position's rank in sorted order
    
    # Select top max_remask candidates (lowest confidence)
    max_remask_exp = max_remask.unsqueeze(1)  # [B, 1]
    remask_index = (ranks < max_remask_exp) & remask_candidates  # [B, L]
    
    # Ensure fix_mask positions are never remasked (double protection)
    if fix_mask is not None:
        remask_index = torch.logical_and(remask_index, ~fix_mask)  # Never remask fixed positions

    # Step 6: Apply remask - set low confidence positions back to mask_id
    x0 = torch.where(remask_index, mask_id, x0)

    # transfer_index: all True, meaning x0 is the final updated block
    transfer_index = torch.ones_like(orig_mask_index, dtype=torch.bool)
    
    return x0, transfer_index


class ThresholdParallelDecoder(ParallelDecoder):
    """Parallel decoding driven by a confidence threshold."""
    def __init__(
            self,
            temperature,
            threshold,
            remasking='low_confidence',
            mask_id=126336,
            eos_id=126081,
            use_float64=False,
            enable_remask=False,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.eos_id = eos_id
        self.use_float64 = use_float64
        # Enable remasking based on the enable_remask parameter
        self.enable_remask = enable_remask

    def block_init(self, block_x, block_id):
        """Initialize fix_mask to protect prompt tokens from remasking.
        
        If remasking is enabled, creates fix_mask to mark prompt tokens (non-mask_id tokens)
        that should be protected from remasking operations.
        """
        if self.enable_remask:
            # Create fix_mask: mark positions that are NOT mask_id (i.e., prompt tokens)
            # These positions should be protected from remasking
            self.fix_mask = (block_x != self.mask_id)  # [B, L], True for prompt tokens
        else:
            self.fix_mask = None

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in the same block of multiple samples.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]

        if self.enable_remask:
            # Get fix_mask for current block (protect prompt tokens from remasking)
            fix_mask = getattr(self, 'fix_mask', None)
            x0, transfer_index = get_transfer_index_threshold_remask(
                logits,
                self.temperature,
                mask_index,
                curr_x,
                self.mask_id,
                threshold=iter_threshold,
                use_float64=self.use_float64,
                fix_mask=fix_mask,
            )
        else:
            x0, transfer_index = get_transfer_index_threshold(
                logits,
                self.temperature,
                mask_index,
                curr_x,
                self.mask_id,
                threshold=iter_threshold,
                use_float64=self.use_float64,
            )
        
        # For remasking case, transfer_index may include remasked positions
        # For non-remasking case, only update masked positions
        if not self.enable_remask:
            transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
        broadcast_if_needed(x.data)

    def batch_decode(self, logits, block_start, x, block_length, iter_threshold=None):
        """ Decode the logits in the different blocks of multiple samples, indicated by 1-d block_start tensor.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        B, T = x.data.shape
        device = x.data.device

        offset = torch.arange(block_length, device=device).unsqueeze(0) + block_start.unsqueeze(1)  # [B, block_length]

        x_block = torch.gather(x.data, 1, offset.clamp(max=T - 1)) 

        mask_index = (x_block == self.mask_id)

        if self.enable_remask:
            # Get fix_mask for current block (protect prompt tokens from remasking)
            fix_mask = getattr(self, 'fix_mask', None)
            x0, transfer_index = get_transfer_index_threshold_remask(
                logits,
                self.temperature,
                mask_index,
                x_block,
                self.mask_id,
                threshold=iter_threshold,
                use_float64=self.use_float64,
                fix_mask=fix_mask,
            )
        else:
            x0, transfer_index = get_transfer_index_threshold(
                logits,
                self.temperature,
                mask_index,
                x_block,
                self.mask_id,
                threshold=iter_threshold,
                use_float64=self.use_float64,
            )

        # For remasking case, transfer_index may include remasked positions
        # For non-remasking case, only update masked positions
        if not self.enable_remask:
            transfer_index = transfer_index & mask_index

        x_updated = torch.where(transfer_index, x0, x_block)

        x_flat = x.data.view(-1)
        flat_idx = offset + torch.arange(B, device=device).unsqueeze(1) * T
        x_flat[flat_idx] = x_updated
        broadcast_if_needed(x.data)


class CreditThresholdParallelDecoder(ThresholdParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold + credit.
    The decoder decodes a token when its confidence is larger than a threshold.
    """
    def __init__(self,
                 credit_alpha=0.7,
                 boost_gamma=0.2,
                 decay_beta=0.8,
                 **kwargs):
        super().__init__(**kwargs)

        self.credit_alpha = credit_alpha
        self.boost_gamma = boost_gamma
        self.decay_beta = decay_beta

        self._credit_mats = {}
        self._credit_iters = {}

    def _apply_credit_fusion(self, logits, mask_index, key):
        """
        EMA-based credit fusion (no CM, no pre-credit):
        - Maintains a per-block CreditMatrix (EMA with decay).
        - Accumulates enhanced top-1 probability only on masked positions.
        - Returns fused_logits.
        """
        B, L, V = logits.shape
        device = logits.device

        mat = self._credit_mats.get(key, None)
        if mat is None or mat.shape != (B, L, V) or mat.device != device:
            mat = torch.zeros((B, L, V), dtype=torch.float32, device=device)
            self._credit_mats[key] = mat
            self._credit_iters[key] = 0

        iter_idx = self._credit_iters[key]

        if iter_idx > 0:
            mat.mul_(self.decay_beta)

        probs = F.softmax(logits.to(torch.float32), dim=-1)
        top1_probs, top1_idx = torch.max(probs, dim=-1)
        enhanced = top1_probs.pow(self.boost_gamma).to(mat.dtype)
        update_vals = enhanced * mask_index.to(enhanced.dtype)
        mat.scatter_add_(2, top1_idx.unsqueeze(-1), update_vals.unsqueeze(-1))

        fused_logits = logits + self.credit_alpha * torch.log(mat + 1)
        self._credit_iters[key] = iter_idx + 1
        return fused_logits

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block."""
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        key = (block_start, block_end)
        used_logits = self._apply_credit_fusion(logits, mask_index, key)

        x0, transfer_index = get_transfer_index_threshold(
            used_logits,
            self.temperature,
            mask_index,
            curr_x,
            self.mask_id,
            threshold=iter_threshold,
            use_float64=self.use_float64,
        )

        transfer_index = torch.logical_and(transfer_index, mask_index)
        assert transfer_index.dtype == torch.bool
        x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)

        if hasattr(x, 'data'):
            has_mask = (x.data == self.mask_id).any()
        else:
            if x.dim() > 0:
                has_mask = (x == self.mask_id).any()
            else:
                has_mask = (x == self.mask_id)

        if not has_mask:
            self._credit_mats.clear()
            self._credit_iters.clear()
        broadcast_if_needed(x.data)


class FixedParallelDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a fixed number of steps."""
    def __init__(
            self,
            temperature,
            steps,
            remasking='low_confidence',
            mask_id=126336,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        # TODO(zhengda) fix steps when distributed version changes gen length.
        block_mask_index = block_x == self.mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, self.steps
        )
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block."""
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(
            logits,
            self.temperature,
            self.remasking,
            mask_index,
            curr_x,
            self.num_transfer_tokens[:, self.iter],
            None,
        )
        self.iter += 1
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        broadcast_if_needed(x.data)


class HierarchyDecoder(ParallelDecoder):
    """Decode tokens hierarchically to force separate decisions."""
    def __init__(
            self,
            temperature,
            remasking='low_confidence',
            mask_id=126336,
            eos_id=126081,
            threshold=None,
            low_threshold=0.4,
    ):
        super().__init__(temperature, remasking, mask_id)
        self.iter = 0
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.threshold = threshold
        self.low_threshold = low_threshold

    def get_transfer_index(self, logits, mask_index, iter_threshold, **kwargs):

        B, L = mask_index.shape

        # TODO(DuLun): support batch size > 1
        assert B == 1

        device = logits.device

        if not math.isclose(self.temperature, 0.0):
            logits_with_noise = add_gumbel_noise(
                logits, temperature=self.temperature
            )
        else:
            logits_with_noise = logits

        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

        x0_logp = F.log_softmax(logits, dim=-1).gather(
            -1, x0.unsqueeze(-1)
        ).squeeze(-1)
        x0_p = x0_logp.exp()  # b, l

        neg_inf_val = torch.finfo(x0_p.dtype).min
        confidence = torch.where(
            mask_index,
            x0_p,
            torch.tensor(neg_inf_val, device=device, dtype=x0_p.dtype),
        )

        prev = torch.cat(
            [
                mask_index.new_zeros((B, 1), dtype=torch.bool),
                mask_index[:, :-1],
            ],
            dim=1,
        )
        starts = torch.logical_and(mask_index, torch.logical_not(prev))

        seg_id = torch.cumsum(starts.to(torch.int64), dim=-1) - 1
        seg_id = torch.where(mask_index, seg_id, 0)

        seg_max = torch.full(
            (B, L), neg_inf_val, device=device, dtype=confidence.dtype
        )
        seg_max = torch.scatter_reduce(
            seg_max,
            dim=1,
            index=seg_id,
            src=confidence,
            reduce='amax',
            include_self=True,
        )

        seg_max_at_pos = seg_max.gather(dim=1, index=seg_id)
        transfer_index = (confidence == seg_max_at_pos)

        if self.low_threshold is not None:
            transfer_index = torch.logical_and(
                transfer_index, torch.gt(confidence, self.low_threshold)
            )
        if iter_threshold is not None:
            transfer_index = torch.logical_or(
                transfer_index, torch.gt(confidence, iter_threshold)
            )

        top1_idx = torch.argmax(confidence, dim=-1)
        top1 = torch.nn.functional.one_hot(
            top1_idx, num_classes=L
        ).to(torch.bool)
        transfer_index = torch.logical_or(transfer_index, top1)

        return x0, transfer_index

    def block_init(self, block_x, block_id):
        # TODO(zhengda) fix steps when distributed version changes gen length.
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold=None):
        """ Decode the logits in a block.
        """
        if iter_threshold is None:
            iter_threshold = self.threshold
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        x0, transfer_index = self.get_transfer_index(
            logits, mask_index, iter_threshold
        )
        self.iter += 1
        transfer_index = torch.logical_and(transfer_index, mask_index)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        broadcast_if_needed(x.data)
