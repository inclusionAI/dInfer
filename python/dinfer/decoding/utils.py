import math
import copy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if math.isclose(temperature, 0.0):
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def calculate_op_num(x, hidden_size=4096, mlp_hidden_size = 12288, vocab_size = 126464, num_hidden_layers=32, cache_length=0):
    cfg_factor = 1
    qkv_ops = 4*x.shape[0]*hidden_size*hidden_size*x.shape[1]*2
    attn_ops = x.shape[0]*(cache_length)*x.shape[1]*hidden_size*2
    ffn_ops = 3*x.shape[0]*hidden_size*mlp_hidden_size*x.shape[1]*2
    layer_ops = qkv_ops + attn_ops + ffn_ops
    op_num = cfg_factor * (num_hidden_layers*layer_ops + x.shape[0]*hidden_size*vocab_size*x.shape[1]*2)
    return op_num/1e12 

class TokenArray:
    """ A token array to support read, update and expansion.

    We need to access the tokens that have been generated and write new tokens to the array.
    Some algorithms require to expand the token array.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device):
        self.prompt = prompt.to(device)
        self.data = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.gen_length = gen_length
        self.eos_id = eos_id
        self.mask_id = mask_id

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def batch_size(self):
        return self.prompt.shape[0]

    @property
    def device(self):
        return self.data.device

    def expand(self, new_len):
        pass

    def get_generated_tokens(self):
        if self.batch_size == 1:
            return self.data[self.data != self.eos_id].unsqueeze(0)
        else:
            self.data[self.data == self.mask_id] = self.eos_id
            return self.data

    def select_seqs(self, idx):
        arr = copy.copy(self)
        arr.prompt = self.prompt[idx]
        arr.data = self.data[idx]
        return arr

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vals):
        self.data[idx] = vals

class DistAlignedTokenArray:
    """ A token array to support read, update and expansion in the distributed setting.

    In this setting, each process still contains the full copy of the token array.
    The main difference from TokenArray is that this class makes sure that the length of the token array
    is rounded to the world size.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    rank : int
        The rank of the process
    world_size : int
        The number of processes.
    """
    def __init__(self, prompt, gen_length, mask_id, eos_id, device, rank, world_size):
        total_length = prompt.shape[1] + gen_length
        if total_length % world_size != 0:
            total_length = (total_length // world_size + 1) * world_size
        self.data = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.orig_gen_length = gen_length
        self.gen_length = total_length - prompt.shape[1]
        self.prompt = prompt
        self.eos_id = eos_id
        self.mask_id = mask_id

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def device(self):
        return self.data.device

    def get_generated_tokens(self):
        return self.data[self.data != self.eos_id].unsqueeze(0)

    def expand(self, new_len):
        pass

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, vals):
        self.data[idx] = vals

class BlockLoc:
    """ The location of the block in the token array.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end



class BlockIterator:
    """ Block iterator

    This performs block-wise iteration on the input token array for diffusion decoding.

    Parameters
    ----------
    x : TokenArray
        The token array that contains decoded tokens and stores the new generated tokens
    block_length : int
        The length of the block
    start_block_align : bool
        Align the first decoding block to the block size. The first block may overlap with the prompt.
    """
    def __init__(self, x, block_length, start_block_align=False):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.start_block_align = start_block_align
        if start_block_align:
            self.first_block_start = self._get_first_block_start()
        else:
            self.first_block_start = self.x.prompt.shape[1]

    def _get_first_block_start(self):
        prompt = self.x.prompt
        non_mask_number = (prompt != self.x.mask_id).sum(dim=-1).min().item()
        start = ((non_mask_number) // self.block_length) * self.block_length
        return start

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        current_block_start = self.first_block_start + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = min(current_block_start + self.block_length, self.x.total_length)
        assert current_block_end <= self.x.total_length
        self.iter += 1
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]

class BlockDiffusionIterator():
    """ Block iterator

    This performs block-wise iteration on the input token array for diffusion decoding.

    Parameters
    ----------
    x : TokenArray
        The token array that contains decoded tokens and stores the new generated tokens
    block_length : int
        The length of the block
    """
    def __init__(self, x, block_length):
        self.x = x
        self.iter = 0
        self.block_length = block_length
        self.first_block_start = self._get_first_block_start()
    
    def _get_first_block_start(self):
        prompt = self.x.prompt
        non_mask_number = (prompt != self.x.mask_id).sum(dim=-1).min().item()
        start = ((non_mask_number) // self.block_length) * self.block_length
        return start

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        current_block_start = self.first_block_start + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = min(current_block_start + self.block_length, self.x.total_length)
        assert current_block_end <= self.x.total_length
        self.iter += 1
        # Fix dimension bug: originally was self.x[current_block_start:current_block_end], 
        # now correctly uses self.x[:, current_block_start:current_block_end] to handle batch dimension
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]


class BlockIteratorFactory:
    """ Iterator factory

    This generates iterators for DiffusionLLM to iterate over a sequence.

    Parameters
    ----------
    start_block_align : bool
        Align the first decoding block to the block size. The first block may overlap with the prompt.
    
    use_block_diffusion: bool
        If this flag set to True, the block diffusion iteration will be used and start_block_algin will be ignored.

    Returns
    -------
    BlockIterator : the block iterator.
    """
    def __init__(self, start_block_align=False, use_block_diffusion=False):
        self._start_block_align = start_block_align
        self._use_bd = use_block_diffusion

    def create(self, x, block_length):
        if self._use_bd:
            return BlockDiffusionIterator(x, block_length)
        else:
            return BlockIterator(x, block_length, start_block_align=self._start_block_align)


class KVCache:
    """ The KV-cache

    Parameters
    ----------
    past_key_values : List[torch.Tensor]
        The keys and values of each transformer layer.
    """
    def __init__(self, past_key_values, backend='vllm', length=2048, cache_align_size=128):
        if backend == 'vllm':
            assert len(past_key_values) % 2 == 0
            self._data = past_key_values
        else:
            self.cache_align_size = cache_align_size
            assert len(past_key_values) % 2 == 0
            self._raw_data = past_key_values
            self._consolidate_raw()

            n = -(-(self._raw_data.shape[4] + 64) // self.cache_align_size)
            next_pow2 = 1 << (n - 1).bit_length() if n > 1 else 1
            self.length = next_pow2 * self.cache_align_size

            device = self._raw_data.device
            num_layer, _, batch_size, num_heads, seq_len, hidden_dim = self._raw_data.shape
            self._data = torch.zeros(num_layer, 2, batch_size, num_heads, self.length, hidden_dim, device=device, dtype=torch.bfloat16)
            self._data[:, :, :, :, :seq_len] = self._raw_data
    def consolidate(self):
        if isinstance(self._data, torch.Tensor):
            return

        num_layers = len(self._data) // 2
        inner_shape = self._data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._data = torch.stack(self._data, dim=0).reshape(num_layers, 2, *inner_shape)

    def _consolidate_raw(self):
        if isinstance(self._raw_data, torch.Tensor):
            return

        num_layers = len(self._raw_data) // 2
        inner_shape = self._raw_data[0].shape
        # The shape is [num_layers, 2, batch_size, num_heads, seq_len, hidden_dim]
        self._raw_data = torch.stack(self._raw_data, dim=0).reshape(num_layers, 2, *inner_shape)

    @property
    def num_layers(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[0]

    @property
    def seq_len(self):
        assert isinstance(self._data, torch.Tensor)
        return self._data.shape[4]

    def get_keys(self, layer_idx):
        """ Get the keys of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][0]

    def get_values(self, layer_idx):
        """ Get the values of a transformer layer.
        """
        assert isinstance(self._data, torch.Tensor)
        return self._data[layer_idx][1]

    def update(self, key_states, val_states, layer_idx, replace_position=None, backend='vllm'):
        """ Update the keys and values of a transformer layer.

        Parameters
        ----------
        key_states : torch.Tensor
            The keys in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        val_states : torch.Tensor
            The values in a block of tokens. The shape is [batch_size, num_heads, seq_len, hidden_dim]
        layer_idx : int
            The index of the transformer layer
        replace_position : Tuple[int]
            The start and the end position where keys and values should be updated.

        Returns
        -------
        torch.Tensor: the new keys for the entire sequence of the transformer layer.
        torch.Tensor: the new values for the entire sequence of the transformer layer.
        """
        if backend ==  'vllm':
            # This is dual cache.
            if replace_position is not None:
                keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=replace_position[0], end=replace_position[1])
                values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=replace_position[0], end=replace_position[1])
            else:
                # This is prefix cache.
                keys = torch.cat([self.get_keys(layer_idx), key_states], dim=2)
                values = torch.cat([self.get_values(layer_idx), val_states], dim=2)
        else:
            cache_length = self.get_keys(layer_idx).shape[2]
            block_length = key_states.shape[2]
            keys = self.get_keys(layer_idx).slice_scatter(key_states, dim=2, start=cache_length - block_length, end=cache_length)
            values = self.get_values(layer_idx).slice_scatter(val_states, dim=2, start=cache_length - block_length, end=cache_length)
        return keys, values


class DiffusionKVCacheManager:
    """ KV-cache for diffusion LLM.

    The KV-cache caches the KV of the tokens before and after the block that is being decoded.
    Because diffusion LLM uses bidirectional attention, the KV-cache has to be updated frequently in the diffusion iterations.
    This class basically defines the KV-cache update policy in the diffusion iterations. This includes the locations where
    keys and values can be updated and the frequency of the keys and values can be updated.

    """
    def __init__(self, cache_update_freq=None, cache_type='prefix', backend='vllm', max_length=2048):
        self.past_key_values = None
        self.block_start = None
        self.block_end = None
        self.cache_update_freq = cache_update_freq
        assert cache_type in ['prefix', 'dual']
        self.cache_type = cache_type
        self.backend=backend
        self.max_length = max_length

    def require_update(self, iter_no, block_start, block_end):
        """ require to update the kv-cache.

        Parameters
        ----------
        iter_no : int
            The diffusion iteration number
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.
        """
        if self.past_key_values is None:
            _require_update = True
        # If self.cache_update_freq is not specified, the KV-cache is updated when we enter a new block.
        if self.cache_update_freq is None:
            _require_update = self.block_start != block_start or self.block_end != block_end
        else:
            # Otherwise, we update the KV-cache when we enter a new block or the specified number of
            # diffusion iterations is reached.
            _require_update = iter_no % self.cache_update_freq == 0 \
                    or (self.block_start != block_start or self.block_end != block_end)
        # TODO(zhengda) change update logic to block idx
        self.block_start = block_start
        self.block_end = block_end
        return _require_update

    def update(self, past_key_values, range_start=None, range_end=None):
        """ update the KV-cache

        Parameters
        ----------
        past_key_values : List[torch.Tensor]
            The key values in all transformer layers.
        range_start : int
            The start of the range that is being updated.
        range_end : int
            The end of the range that is being updated.
        """
        if isinstance(past_key_values, KVCache):
            self.past_key_values = past_key_values
        else:
            self.past_key_values = KVCache(past_key_values, self.backend, self.max_length)
        # We should make sure the kv-cache in all layers are converted into a tensor.
        self.past_key_values.consolidate()
        

    def range_update(self, past_key_values, range_start=0, range_end=0, block_length=32):
        """ update the KV-cache

        Parameters
        ----------
        past_key_values : List[torch.Tensor]
            The key values in all transformer layers.
        range_start : int
            The start of the range that is being updated.
        range_end : int
            The end of the range that is being updated.
        """
        if isinstance(past_key_values, KVCache):
            # raise ValueError("past_key_values should be a list of tensors")
            self.past_key_values = past_key_values
        else:
            num_layers = len(past_key_values) // 2
            inner_shape = past_key_values[0].shape
            compact_kvcache = torch.stack(past_key_values, dim=0).reshape(num_layers, 2, *inner_shape)
            if block_length>0:
                self.past_key_values = KVCache(torch.cat((compact_kvcache[:, :, :, :, range_start:range_end-block_length], compact_kvcache[:, :, :, :, -block_length:]), dim=4), self.backend, self.max_length)
            else:
                self.past_key_values = KVCache(compact_kvcache[:, :, :, :, range_start:range_end], self.backend, self.max_length)
        # We should make sure the kv-cache in all layers are converted into a tensor.
        self.past_key_values.consolidate()
        # print(self.past_key_values._data.shape)

    def get_key_values(self, block_start, block_end):
        """ Get the key-values given the block that is being decoded.

        Parameters
        ----------
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.

        Returns
        -------
        List[List[torch.Tensor]] : the key-values required to decode the specified block.
        torch.Tensor : the tensor indicates the valid locations in the returned key-values.
        """
        # The key-value cache cannot be empty.
        assert self.past_key_values is not None
        if self.cache_type == 'prefix':
            replace_position = (int(block_start), int(self.past_key_values.seq_len))
        else:
            replace_position = (int(block_start), int(block_end))
        return self.past_key_values, replace_position

class BlockDiffusionPrefixCacheManager(DiffusionKVCacheManager):
    """
     KVcache manager of block diffusion.
    """

    def get_key_values(self, block_start, block_end):
        # use prefix cache for block diffusion.
        return self.past_key_values, (int(block_start), int(block_end))

    def extend_cache(self, end):
        """
        When move to new block, extend the kvcache length from previous block end to new block end location.

        Parameters
        ----------
        length : int
            The extended length (equvelent to block length)
        
        """
        if self.backend == 'vllm':
            cur_kv_length = self.past_key_values._data.shape[-2]
            extended_cache = F.pad(self.past_key_values._data, pad=(0, 0, 0, end-cur_kv_length), mode='constant', value=0)
            self.past_key_values._data = extended_cache
        else:
            cur_kv_length = self.past_key_values._data.shape[-2]
            n = -(-end // self.past_key_values.cache_align_size)  # 等价于 ceil(target / cache_align_size)，纯整数
            next_pow2 = 1 << (n - 1).bit_length() if n > 1 else 1
            aligned_end = next_pow2 * self.past_key_values.cache_align_size

            if aligned_end <= cur_kv_length:
                return
            extended_cache = F.pad(self.past_key_values._data, pad=(0, 0, 0, aligned_end-cur_kv_length), mode='constant', value=0)
            self.past_key_values._data = extended_cache


class KVCacheFactory:
    """ KV-cache factory.

    This class generates KV-cache for the diffusion LLM when it runs diffusion iterations.
    """
    def __init__(self, cache_type, cache_update_freq=None, is_bd_model=False, backend='vllm', max_length=2048):
        self.cache_type = cache_type
        self.cache_update_freq = cache_update_freq
        self.is_bd_model = is_bd_model
        self.backend = backend
        self.max_length = max_length

    def create(self):
        if self.is_bd_model:
            return BlockDiffusionPrefixCacheManager(cache_update_freq=self.cache_update_freq, cache_type=self.cache_type, backend=self.backend, max_length=self.max_length)
        else:
            return DiffusionKVCacheManager(cache_update_freq=self.cache_update_freq, cache_type=self.cache_type)

def gather_sequence_block(partial_data, partial_start, partial_end, block_start, block_end, rank, world_size):
    """ Gather the wanted block data from the partitioned data.

    Each process contains a partition specified by `partial_start` and `partial_end`.
    The wanted block is located between `block_start` and `block_end`.

    We want to gather the data within the block range from the partitioned data.
    """
    if partial_start >= block_end or partial_end <= block_start:
        # there is no overlap, nothing is needed from partial_data
        arr = partial_data[:, 0:0]
    elif block_start >= partial_start and block_end <= partial_end:
        # the needed block is within partial_data.
        arr = partial_data[:, (block_start - partial_start):(block_end - partial_start)]
    elif block_start <= partial_start and block_end >= partial_end:
        # the needed partition is within the block.
        arr = partial_data
    elif partial_start >= block_start and partial_end >= block_end:
        # the needed block is overlapped in the front of partial_data
        arr = partial_data[:, 0:(block_end - partial_start)]
    else:
        # the needed block is overlapped at the end of partial_data
        arr = partial_data[:, (block_start - partial_start):(partial_end - partial_start)]
    arr = arr.contiguous()

    shape_list = [
            torch.zeros(len(arr.shape), dtype=torch.int64, device=partial_data.device) for _ in range(world_size)
    ]
    dist.all_gather(shape_list, torch.tensor(arr.shape, dtype=torch.int64, device=partial_data.device))
    part_list = [
            torch.zeros(*tuple(shape.tolist()), dtype=partial_data.dtype, device=partial_data.device) for shape in shape_list
    ]
    dist.all_gather(part_list, arr)
    return torch.cat(part_list, dim=1)
