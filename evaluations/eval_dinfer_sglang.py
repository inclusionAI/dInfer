'''
This file is inspired by the code from https://github.com/NVlabs/Fast-dLLM
'''
import accelerate
import torch
import random
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm, trange
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
from multiprocessing import Process
from lm_eval.api.model import LM
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer.model import LLaDAMoeModelLM, LLaDAModelLM, LLaDA2MoeModelLM
from dinfer import BlockIteratorFactory, KVCacheFactory
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM, BlockDiffusionLLM    
from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.moe import initialize_moe_config
from dataclasses import dataclass


datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
datasets.config.DOWNLOAD_TIMEOUT = 180 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


bucket_size = 32
used_buckets = []

def cut_eos(data, eos_id=156892):
    eos_indices = (data[0] == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        first_eos_idx = eos_indices[0].item()
        return data[:, :first_eos_idx]
    else:
        return data

@ torch.no_grad()
def run_benchmark(world_size, rank, gpu_id, tokenizer, args):
    print('started', world_size, rank, gpu_id)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    all_input_ids, padded_gen_lens = args.all_input_ids, args.padded_gen_lens

    block_length=args.block_length
    mask_id = 156895
    eos_id = 156892

    from sglang.srt import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.master_port)
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(args.tp_size, args.tp_size, 1, backend='nccl')
    print("[Loading model]")

    from sglang.srt.layers.dp_attention import initialize_dp_attention
    model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    server_args = ServerArgs(model_path=args.model_name, enable_dp_attention=True, trust_remote_code=True, tp_size=args.tp_size, dp_size = 1, pp_size = 1)
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
    except ImportError:
        pass
    else:
        set_global_server_args_for_scheduler(server_args)
    initialize_dp_attention(
        server_args=server_args,
        model_config=model_config,
    )
    initialize_moe_config(server_args)
    model = LLaDA2SGLangLM(config=model_config, expert_map_path='.').eval()
    torch.set_default_dtype(torch.bfloat16)
    model.load_weights(args.model_name, device=device)
    initialize_moe_config(server_args)

    model = model.to(device)
    input_lengths = [inp.size(-1) for inp in all_input_ids]
    max_length = max(input_lengths)+args.gen_len
    model = ModelRunner(model, device, server_args=server_args, max_length=max_length)
    
    batch_size = args.batch_size

    if args.parallel_decoding == 'threshold':
        if args.use_credit:
            decoder = CreditThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id, enable_remask=args.enable_remask)
        else:
            decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=mask_id, eos_id=eos_id, enable_remask=args.enable_remask)

    else:
        decoder = HierarchyDecoder(temperature=0, threshold=args.threshold, low_threshold=args.low_threshold, mask_id=mask_id, eos_id=eos_id)

    use_sw = args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0

    if args.cache == 'prefix' or args.cache == 'dual':
        cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd, backend='sglang', max_length=max_length)
        # cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd)

    else:
        cache_factory=None

    if not args.use_bd:
        if args.cont_weight>0:
            if use_sw:
                print("IterSmoothWithVicinityCacheDiffusionLLM")
                dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                    cont_weight=args.cont_weight, prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                print("IterSmoothDiffusionLLM")
                dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight)
        else:
            if use_sw:
                print("VicinityCacheDiffusionLLM")
                dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                print("BlockWiseDiffusionLLM")
                dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, use_shift=args.use_shift)
    else:
        print("BlockDiffusionLLM")
        dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True), cache_factory=cache_factory, early_stop=True, maximum_unroll=4, expected_tpf=4, backend='sglang')
        # dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True)

    
            
    input_lengths = [inp.size(-1) for inp in all_input_ids]
    sorted_indices = sorted(range(len(input_lengths)), key=lambda i: input_lengths[i])

    sorted_input_ids = [all_input_ids[i] for i in sorted_indices]
    sorted_padded_gen_lens = [padded_gen_lens[i] for i in sorted_indices]

    for wi in range(1):
        outputs = []
        total_forward = 0
        if rank==0:
            iterator = trange(0, len(sorted_input_ids), batch_size)
        else:
            iterator = range(0, len(sorted_input_ids), batch_size)
        start = time.time()
        tpfs = []
        tpss = []
        fpss = []
        total_token = 0
        token_numbers = []
        total_time = 0
        for i in iterator:   
            input_ids = sorted_input_ids[i:i+batch_size]

            prefill_blocks = input_ids[-1].shape[1] // block_length
            prefill_length = prefill_blocks * block_length

            max_length = input_ids[-1].shape[1]
            min_padded_length = sorted_padded_gen_lens[i+len(input_ids)-1]
            batch_input_ids= torch.zeros((len(input_ids), max_length), dtype=torch.long, device=device).fill_(156895)
            for j in range(len(input_ids)):
                batch_input_ids[j, :input_ids[j].shape[1]] = input_ids[j].to(device)
            input_ids = batch_input_ids
            inner_start = time.time()
            prev_forwards = dllm.num_forwards
            out = dllm.generate(input_ids, gen_length=min_padded_length, block_length=block_length)
            nfe = dllm.num_forwards - prev_forwards
            inner_stop = time.time()
            sample_time = inner_stop - inner_start
            for j in range(input_ids.shape[0]):
                outputs.append(out[j].unsqueeze(0))
            total_forward += nfe
            total_time += sample_time
            batch_token_number = 0
            for j in range(input_ids.shape[0]):
                token_number = int((out[j]!=156892).sum() - sorted_input_ids[i+j].shape[1])
                batch_token_number += token_number
                token_numbers.append(token_number)
            tpf = batch_token_number/nfe/batch_size
            tps = batch_token_number/sample_time
            fps = nfe/sample_time
            tpfs.append(tpf)
            tpss.append(tps)
            fpss.append(fps)
            if rank == 0:
                print(f'[iter {i:4d}]nfe={nfe:4d}, token number={batch_token_number:4d}, sample_time={sample_time:2.4f}, fps={fps:4.2f}({np.mean(fpss):4.2f}),tpf={tpf:2.2f}({np.mean(tpfs):4.2f}), tps={tps:4.2f}({np.mean(tpss):4.2f})')
                if wi==0 and i<5:
                    for j in range(min(input_ids.shape[0], 4)):
                        answer = cut_eos(out[j, sorted_input_ids[i+j].shape[1]:].unsqueeze(0))[0]
                        # print(answer)
                        print(f'generated text {j}: {tokenizer.decode(answer, skip_special_tokens=False)}')
            total_token += token_number

        total_token = total_token

        stop = time.time()


    original_order_outputs = [None] * len(all_input_ids)
    original_order_tpfs = [None] * len(all_input_ids)
    original_order_tpss = [None] * len(all_input_ids)
    original_order_fpss = [None] * len(all_input_ids)
    original_order_token_numbers = [None] * len(all_input_ids)

    for i, original_idx in enumerate(sorted_indices):
        original_order_outputs[original_idx] = outputs[i//batch_size]
        original_order_tpfs[original_idx] = tpfs[i//batch_size]
        original_order_tpss[original_idx] = tpss[i//batch_size]
        original_order_fpss[original_idx] = fpss[i//batch_size]
        original_order_token_numbers[original_idx] = token_numbers[i//batch_size]

    outputs = original_order_outputs
    tpfs = original_order_tpfs
    tpss = original_order_tpss
    fpss = original_order_fpss
    token_numbers = original_order_token_numbers        

    if rank==0:
        answers = []
        for i in trange(len(outputs)):
            out = outputs[i]
            answer = (tokenizer.decode(out[0, all_input_ids[i].shape[1]:], skip_special_tokens=True))
            answers.append(answer)
        print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/total_time}({np.mean(fpss)}), TPS: {total_token/total_time}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')
        filename = args.save_path
        with open (filename, 'w') as f:
            for i in range(len(answers)):
                answer = answers[i]
                json.dump({'answer': answer, 'generated_length': token_numbers[i], 'tpf':tpfs[i//batch_size], 'tps':tpss[i//batch_size], 'fps':fpss[i//batch_size], }, f)
                f.write('\n')
        with open(args.speed_path, 'a+') as f:
            print( args.config, args.parallel_decoding, args.threshold, args.prefix_look, args.batch_size, args.block_length, total_forward, stop-start, total_token / len(all_input_ids), total_forward/total_time, total_token/total_time, total_token/total_forward, sum(padded_gen_lens)/total_forward, np.mean(fpss), np.mean(tpss), np.mean(tpfs), file=f)


@dataclass
class EvalConfig:
    model_name: str = ''
    gpu: str = '0;1;2;3'
    batch_size: int = 1
    gen_len: int = 1024
    prefix_look: int = 0
    after_look: int = 0
    block_length: int = 64
    threshold: float = 0.9
    warmup_times: int = 0
    low_threshold: float = 0.3
    cont_weight: float = 0
    parallel_decoding: str = 'threshold'
    use_credit: bool = False
    cache: str = ''
    use_tp: bool = False
    save_path: str = ''
    config: int = 0
    tp_size: int = 1
    port_offset: int = 0
    all_input_ids = None
    padded_gen_lens = None
    use_cudagraph: bool = False
    use_compile: bool = True
    use_bd: bool = False
    use_shift: bool = False
    model_type: str = 'llada'
    vocab_size: int = 156896
    master_port: int = 23456
    batch_size: int = 1
    save_samples: bool = False
    speed_path: str = ''
    enable_remask: bool = False

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@register_model("dInfer_eval")
class DInferEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        device="cuda",
        mask_id=126336,
        eos_id=126081,
        max_length=4096,
        batch_size=2,
        mc_num=128,
        is_check_greedy=True,
        gen_length=1024,
        block_length=1024,
        save_dir=None,
        show_speed=False,
        parallel_decoding="threshold",
        threshold: float=0.9,
        cache: str="",
        warmup_times: int=0,
        low_threshold: float=0.3,
        cont_weight: float=0,
        use_credit: bool=False,
        tp_size: int=1,
        parallel = 'dp',
        use_compile = True,
        master_port = 23456,
        use_cudagraph = True,
        gpus = '0;1;2;3',
        use_bd = False,
        prefix_look = 0,
        after_look = 0,
        use_shift = False,
        model_type = 'llada2',
        save_samples = False,
        enable_remask = False,
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
        self.gen_length = gen_length
        self.block_length = block_length
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.parallel_decoding = parallel_decoding
        self.threshold = threshold
        self.cache = cache
        self.warmup_times = warmup_times
        self.low_threshold = low_threshold
        self.cont_weight = cont_weight
        self.use_credit = use_credit
        self.master_port = master_port
        self.tp_size = tp_size
        self.use_compile = use_compile
        self.parallel = parallel
        self.use_cudagraph = use_cudagraph
        self.gpus = gpus
        self.prefix_look = prefix_look
        self.after_look = after_look
        self.use_bd = use_bd
        self.kwargs = kwargs
        self.use_shift = use_shift
        self.model_type = model_type
        self.save_samples = save_samples
        self.enable_remask = enable_remask

        if self.model_type == 'llada2': 
            self.mask_id = 156895
            self.eos_id = 156892
            self.vocab_size = 156896
            self.is_moe = True
        else:
            raise ValueError('model type not supported')

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
        
            
        if parallel == 'tp':
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
            self.save_path = os.path.join(self.save_dir, f'rank_{self.rank}.jsonl')
            print(f"save_path: {self.save_path}")
            self.speed_path = os.path.join(self.save_dir, f'results.txt')

        

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

        all_input_ids = load_inputs(requests, self.tokenizer)
        padded_gen_lens = cal_bucket_len(self.gen_length, all_input_ids)
    
        procs = []
        answers = []
        gpus = [int(gpu) for gpu in self.gpus.split(';')]
        args = {"gpu": gpus, "batch_size": self.batch_size, "model_name": self.model_path, "gen_len": self.gen_length, "block_length": self.block_length, "prefix_look": self.prefix_look, "after_look": self.after_look, "warmup_times": self.warmup_times, "low_threshold": self.low_threshold, "threshold": self.threshold, "cont_weight": self.cont_weight, "use_credit": self.use_credit, "cache": self.cache, "parallel_decoding": self.parallel_decoding, "tp_size": self.tp_size, "save_path": self.save_path, "use_cudagraph": self.use_cudagraph, "use_compile": self.use_compile,"use_bd": self.use_bd, "use_shift": self.use_shift, "model_type": self.model_type, "vocab_size": self.vocab_size, "batch_size": self.batch_size, "speed_path": self.speed_path, "enable_remask": self.enable_remask}
        args = EvalConfig(**args)
        args.tp_size = len(gpus)
        args.master_port = self.master_port
        args.use_tp = args.tp_size > 1
        args.port_offset = gpus[0]
        args.all_input_ids = all_input_ids
        args.padded_gen_lens = padded_gen_lens
        
        if len(gpus) == 1:
            run_benchmark(1, 0, gpus[0], self.tokenizer, args)
        else:
            for i, gpu in enumerate(gpus):
                ctx = mp.get_context('spawn')
                p = ctx.Process(target=run_benchmark, args=(len(gpus), i, gpu, self.tokenizer, args))
                # p.daemon = True
                procs.append(p)
                p.start()
            for p in procs:
                p.join()
        answers = []
        with open(self.save_path, 'r') as f:
            for line in f :
                answers.append(json.loads(line)["answer"])
        if not self.save_samples is None:
            os.remove(self.save_path)
        return answers


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()