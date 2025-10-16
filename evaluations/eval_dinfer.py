'''
This file is inspired by the code from https://github.com/NVlabs/Fast-dLLM
'''
import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
import accelerate
import random
import numpy as np
import json
import time
import datasets
import json
import time
import datasets
import os
from transformers import AutoTokenizer, AutoConfig
import torch.multiprocessing as mp
from lm_eval.api.model import LM
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from dinfer.model import FusedOlmoeForCausalLM, LLaDAModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, BlockWiseDiffusionLLMCont, SlidingWindowDiffusionLLM, SlidingWindowDiffusionLLMCont
from dinfer import DiffusionLLMServing, SamplingParams
from vllm import distributed
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.config import ParallelConfig

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
datasets.config.DOWNLOAD_TIMEOUT = 180 

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

bucket_size = 8
used_buckets = []

def get_bucket_length(length):
    bucket_length = bucket_size*(length//bucket_size)
    if bucket_length not in used_buckets:
        used_buckets.append(bucket_length)
    return bucket_length


def load_inputs(prompts, tokenizer):
    all_input_ids = []
    for id, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt.args[0])['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        all_input_ids.append(input_ids)
    return all_input_ids

def cal_bucket_len(gen_len, all_input_ids):
    max_prompt_length = 0
    padded_gen_lens = []

    for i in range(len(all_input_ids)):
        input_ids = all_input_ids[i]
        if input_ids.shape[1] > max_prompt_length:
            max_prompt_length = input_ids.shape[1]
        padded_length = get_bucket_length(input_ids.shape[1]+gen_len)
        padded_gen_lens.append(padded_length - input_ids.shape[1])
    return padded_gen_lens

@register_model("dInfer_eval")
class DInferEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        eos_id=126081,
        max_length=4096,
        batch_size=1,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        device="cuda",
        save_dir=None,
        show_speed=False,
        parallel_decoding="threshold",
        threshold: float=0.9,
        cache: str="",
        prefix_look: int=0,
        after_look: int=0,
        warmup_times: int=0,
        low_threshold: float=0.3,
        cont_weight: float=0,
        use_credit: bool=False,
        tp_size: int=1,
        parallel = 'dp',
        use_compile = True,
        master_port = '23456',
        use_cudagraph = False,
        **kwargs
    ):

        super().__init__()
        
        self.model_path = model_path
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.parallel_decoding = parallel_decoding
        self.threshold = threshold
        self.cache = cache
        self.prefix_look = prefix_look
        self.after_look = after_look
        self.warmup_times = warmup_times
        self.low_threshold = low_threshold
        self.cont_weight = cont_weight
        self.use_credit = use_credit
        self.master_port = master_port
        self.tp_size = tp_size
        self.use_compile = use_compile
        self.parallel = parallel
        self.use_cudagraph = use_cudagraph
        self.kwargs = kwargs

        if "moe" in model_path:
            self.mask_id = 156895
            self.eos_id = 156892

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerate.Accelerator()
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})  
        
        # set decoder
        if parallel_decoding == "threshold":
          if use_credit:
              decoder = CreditThresholdParallelDecoder(temperature=0, threshold=threshold, mask_id=self.mask_id, eos_id=self.eos_id)
          else:
              decoder = ThresholdParallelDecoder(temperature=0, threshold=threshold, mask_id=self.mask_id, eos_id=self.eos_id)
        else:
            decoder = HierarchyDecoder(temperature=0, threshold=threshold, low_threshold=low_threshold,
                                      mask_id=self.mask_id, eos_id=self.eos_id)
        if parallel == 'dp':
            self.device= torch.device(device)
            if "moe" in model_path:
                # initialize tensor parallel but don't use it
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12346'+str(self.rank)
                distributed.init_distributed_environment(1, 0, 'env://', 0, 'nccl')
                distributed.initialize_model_parallel(1, backend='nccl')
                parallel_config = ParallelConfig(enable_expert_parallel = True)
                with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
                    vllm_config = get_current_vllm_config()
                    print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)
                    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                    config.flash_attention = True
                    # load model
                    self.model = FusedOlmoeForCausalLM(config=config).eval()
                    self.model.load_weights(self.model_path, torch_dtype=torch.bfloat16)
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    self.vllm_config = vllm_config
            else:
                self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, **model_kwargs).eval()

            
            if self.accelerator is not None:
                self.model = self.accelerator.prepare(self.model)
                self.device = torch.device(f'{self.accelerator.device}')
            else:
                self.model = self.model.to(self.device)

            if self.use_compile:
                # compile model
                if self.use_cudagraph:
                    self.model.forward = torch.compile(self.model.forward, fullgraph=True, dynamic=False, mode='reduce-overhead')
                else:
                    self.model.forward = torch.compile(self.model.forward, fullgraph=False, dynamic=True)
            
            use_sw = prefix_look > 0 or after_look > 0 or warmup_times > 0
            cache_factory = KVCacheFactory(cache) if cache in ["prefix", "dual"] else None

            if cont_weight > 0:
                if use_sw:
                    self.dllm = SlidingWindowDiffusionLLMCont(self.model, decoder, BlockIteratorFactory(start_block_align=True),
                                                        cache_factory=cache_factory, early_stop=True,
                                                        cont_weight=cont_weight, prefix_look=prefix_look,
                                                        after_look=after_look, warmup_steps=warmup_times)
                else:
                    self.dllm = BlockWiseDiffusionLLMCont(self.model, decoder, BlockIteratorFactory(start_block_align=True),
                                                    cache_factory=cache_factory, early_stop=True,
                                                    cont_weight=cont_weight)
            else:

                if use_sw:
                    self.dllm = SlidingWindowDiffusionLLM(self.model, decoder, BlockIteratorFactory(start_block_align=True),
                                                    cache_factory=cache_factory, early_stop=True,
                                                    prefix_look=prefix_look, after_look=after_look,
                                                    warmup_steps=warmup_times)
                else:
                    self.dllm = BlockWiseDiffusionLLM(self.model, decoder, BlockIteratorFactory(start_block_align=True),
                                                cache_factory=cache_factory, early_stop=True)
        elif parallel == 'tp':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.spwan_ctx = mp.get_context('spawn')
            self.device= torch.device(device)
            params = SamplingParams(parallel_decoding=self.parallel_decoding, temperature=0, threshold=self.threshold, low_threshold=self.low_threshold, mask_id=self.mask_id, eos_id=self.eos_id, 
                                    early_stop=True, cache=self.cache, cont_weight=self.cont_weight, use_credit=self.use_credit, enable_torch_compile=self.use_compile, use_cudagraph=self.use_cudagraph)
            self.dllm = DiffusionLLMServing(model=self.model_path, is_moe=True, sample_params=params, server_port=12355, num_gpus=self.tp_size)

        else:
            raise NotImplementedError(parallel)
        
        

    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @property
    def tokenizer_name(self) -> str:
        return self.model_path
    
    def apply_chat_template(self, chat_history, **kwargs) -> str:
        if "tokenize" not in kwargs:
            kwargs["tokenize"] = False
        return self.tokenizer.apply_chat_template(chat_history, **kwargs)

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    
    def generate_until(self, requests):
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
    
        all_input_ids = load_inputs(requests, self.tokenizer)
        padded_gen_lens = cal_bucket_len(self.gen_length, all_input_ids)
        
        answers = []
        outputs = []
        total_forward = 0
        start = time.time()
        tpfs = []
        tpss = []
        fpss = []
        total_token = 0
        token_numbers = []
        for i, req in enumerate(tqdm(requests, desc="Generating...")):
            input_ids = all_input_ids[i]
            padded_gen_len = padded_gen_lens[i]
            inner_start = time.time()
            if self.parallel == 'dp':
                with set_current_vllm_config(self.vllm_config):
                    input_ids = input_ids.to(self.device)
                    prev_forwards = self.dllm.num_forwards
                    out = self.dllm.generate(input_ids, gen_length=padded_gen_len, block_length=self.block_length)
                    nfe = self.dllm.num_forwards - prev_forwards
            elif self.parallel == 'tp':
                out, nfe = self.dllm.generate(input_ids, gen_length=self.gen_length, block_length=self.block_length)
            else:
                raise NotImplementedError(self.parallel)
            inner_stop = time.time()
            sample_time = inner_stop - inner_start
            outputs.append(out)
            answer = (self.tokenizer.decode(out[0, all_input_ids[i].shape[1]:], skip_special_tokens=True))
            answers.append(answer)
            total_forward += nfe
            token_number = out.shape[1] - input_ids.shape[1]
            token_numbers.append(token_number)
            tpf = token_number/nfe
            tps = token_number/sample_time
            fps = nfe/sample_time
            if self.rank == 0:
                print(f'iter={i}, fps={fps}, nfe={nfe}')
            tpfs.append(tpf)
            tpss.append(tps)
            fpss.append(fps)
            total_token += token_number

        total_token = total_token

        stop = time.time()
        print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/(stop-start)}({np.mean(fpss)}), TPS: {total_token/(stop-start)}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')

        if self.show_speed and self.save_dir is not None:
            with open (self.save_dir+f'/rank{self.rank}_afcpt.jsonl', 'w', encoding='utf-8') as file:
                data={'rank':f'rank{self.rank}',
                      'forward per second': np.mean(fpss),
                      'tokens per second': np.mean(tpss),
                      'tokens per forward': np.mean(tpfs),
                      'average generated length': total_token / len(all_input_ids)
                      }
                file.write(json.dumps(data, ensure_ascii=False) + '\n')
        return answers


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    