import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.distributed as dist
import time
import tqdm
import json
import random

from sglang.srt.server_args import ServerArgs
from sglang.srt.layers.moe import initialize_moe_config
from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
from dinfer.decoding.diffusion_runner import ModelRunner
from dinfer import BlockIteratorFactory, KVCacheFactory, BlockDiffusionLLM
from dinfer import ThresholdParallelDecoder,CreditThresholdParallelDecoder, HierarchyDecoder, BlockWiseDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, IterSmoothWithVicinityCacheDiffusionLLM

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

bucket_size = 32
used_buckets = []

def get_bucket_length(length):
    #bucket_length = bucket_size*((length+bucket_size-1)//bucket_size)
    bucket_length = bucket_size*(length//bucket_size)
    if bucket_length not in used_buckets:
        used_buckets.append(bucket_length)
    return bucket_length

def load_inputs(dataset, tokenizer):
    with open(dataset, 'r') as f:
        data = json.load(f)
    prompts = []
    questions = []
    ids = []
    all_input_ids = []
    if "judge_details" in data.keys():
        details_data = data['judge_details']
    else:
        details_data = data['details']
    for id, judge_detail in enumerate(details_data):
        ids.append(id)
        prompt = judge_detail['prompt']
        questions.append(prompt)
        prompt = '<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>'+prompt+'<|role_end|><role>ASSISTANT</role>'   
        prompts.append(prompt)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        all_input_ids.append(input_ids)
    return all_input_ids, prompts, questions, ids

def cal_bucket_len(args, all_input_ids):
    max_prompt_length = 0
    gen_len = args.gen_len
    padded_gen_lens = []

    for i in range(len(all_input_ids)):
        input_ids = all_input_ids[i]
        if input_ids.shape[1] > max_prompt_length:
            max_prompt_length = input_ids.shape[1]
        padded_length = get_bucket_length(input_ids.shape[1]+gen_len)
        padded_gen_lens.append(padded_length - input_ids.shape[1])
    return padded_gen_lens

def cut_eos(data, eos_id=156892):
    eos_indices = (data[0] == eos_id).nonzero(as_tuple=True)[0]
    if eos_indices.numel() > 0:
        first_eos_idx = eos_indices[0].item()
        return data[:, :first_eos_idx]
    else:
        return data

@ torch.no_grad()
def main(world_size, rank, gpu_id, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    all_input_ids, prompts, questions, ids = load_inputs(args.dataset, tokenizer)
    padded_gen_lens = cal_bucket_len(args, all_input_ids)

    block_length=args.block_length
    dataset_name = args.dataset.split('/')[-1][:-5]
    os.makedirs(args.output_dir, exist_ok=True)

    from sglang.srt import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
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


    if args.parallel_decoding == 'threshold':
        if args.use_credit:
            decoder = CreditThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=156895, eos_id=156892)
        else:
            decoder = ThresholdParallelDecoder(temperature=0, threshold=args.threshold, mask_id=156895, eos_id=156892)

    else:
        decoder = HierarchyDecoder(temperature=0, threshold=args.threshold, low_threshold=args.low_threshold, mask_id=156895, eos_id=156892)
    use_sw = args.prefix_look > 0 or args.after_look > 0 or args.warmup_times > 0
    if args.cache == 'prefix' or args.cache == 'dual':
        cache_factory=KVCacheFactory(args.cache, is_bd_model=args.use_bd, backend='sglang', max_length=max_length)
    else:
        cache_factory=None

    if not args.use_bd:
        if args.cont_weight>0:
            if use_sw:
                dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                    cont_weight=args.cont_weight, prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, cont_weight=args.cont_weight)
        else:
            if use_sw:
                dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True,
                    prefix_look=args.prefix_look, after_look=args.after_look, warmup_steps=args.warmup_times)
            else:
                dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True), cache_factory=cache_factory, early_stop=True, use_shift=args.use_shift)
    else:
        dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True), cache_factory=cache_factory, early_stop=True, maximum_unroll=4, expected_tpf=4, backend='sglang')

    batch_size = args.batch_size
    

    input_lengths = [inp.size(-1) for inp in all_input_ids]
    sorted_indices = sorted(range(len(input_lengths)), key=lambda i: input_lengths[i])

    sorted_input_ids = [all_input_ids[i] for i in sorted_indices]
    sorted_padded_gen_lens = [padded_gen_lens[i] for i in sorted_indices]

    for wi in range(1):
        outputs = []
        total_forward = 0
        if rank==0:
            iterator = tqdm.trange(0, len(sorted_input_ids), batch_size)
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
        for i in tqdm.trange(len(outputs)):
            out = outputs[i]
            answer = (tokenizer.decode(out[0, all_input_ids[i].shape[1]:], skip_special_tokens=True))
            answers.append(answer)
        print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/total_time}({np.mean(fpss)}), TPS: {total_token/total_time}({np.mean(tpss)}), TPF: {total_token/total_forward}({np.mean(tpfs)})')
        filename = args.output_dir+'/'+'_'.join([str(item) for item in [args.exp_name, dataset_name, args.config, args.parallel_decoding, args.threshold, args.prefix_look]])+'.jsonl'
        with open (filename, 'w') as f:
            for i in range(len(answers)):
                question = questions[i]
                prompt = prompts[i]
                answer = answers[i]
                id = ids[i]
                json.dump({'id':id, 'question':question, 'prompt':prompt, 'answer': answer, 'generated_length': token_numbers[i], 'tpf':tpfs[i//batch_size], 'tps':tpss[i//batch_size], 'fps':fpss[i//batch_size], }, f, indent=4)
                f.write('\n')
        with open('results.txt', 'a+') as f:
            print(args.exp_name, args.config, args.parallel_decoding, args.threshold, args.prefix_look, args.batch_size, args.block_length, args.gpu, total_forward, stop-start, total_token / len(all_input_ids), total_forward/total_time, total_token/total_time, total_token/total_forward, sum(padded_gen_lens)/total_forward, np.mean(fpss), np.mean(tpss), np.mean(tpfs), args.dataset, file=f)

    
from multiprocessing import Process
import argparse

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gen_len', type=int, default=1024)
    parser.add_argument('--prefix_look', type=int, default=0)
    parser.add_argument('--after_look', type=int, default=0)
    parser.add_argument('--block_length', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--warmup_times', type=int, default=0)
    parser.add_argument('--low_threshold', type=float, default=0.3)
    parser.add_argument('--cont_weight', type=float, default=0)
    parser.add_argument('--parallel_decoding', type=str, default='hierarchy_faster')
    parser.add_argument('--use_credit', action='store_true')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--use_tp', action='store_true')
    parser.add_argument('--output_dir', type=str, default='/ossfs/workspace/detailed_results_0917')
    parser.add_argument('--use_shift', action='store_true')
    parser.add_argument('--use_bd', action='store_true')
    parser.add_argument('--model_type', type=str, default='mini')
    parser.add_argument('--config', type=int, default=0)
    args = parser.parse_args()
    port = random.randint(40000, 60000)
    args.port = str(port)

    if args.config == 1:
        args.cache = ''
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
    elif args.config == 2:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
    elif args.config == 3:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.95
        args.warmup_times = 4
    elif args.config == 4:
        args.cache = ''
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.8
        args.warmup_times = 0
    elif args.config == 5:
        args.cache = ''
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.8
        args.low_threshold = 0.5
        args.warmup_times = 0
    elif args.config == 6:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.low_threshold = 0.5
        args.warmup_times = 4
    elif args.config == 9:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.9
        args.low_threshold = 0.7
        args.warmup_times = 4

    elif args.config == 10:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.warmup_times = 4
    elif args.config == 11:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.low_threshold = 0.75
        args.warmup_times = 4

    elif args.config == 12:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.low_threshold = 0.5
        args.warmup_times = 4
        
    elif args.config == 13:
        args.cache = 'dual'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.8
        args.warmup_times = 4

    elif args.config == 14:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.9
        args.low_threshold = 0.7
        args.warmup_times = 4

    elif args.config == 15:
        args.cache = 'dual'
        args.parallel_decoding = 'hierarchy_faster'
        args.prefix_look = 16
        args.after_look = 16
        args.threshold = 0.85
        args.low_threshold = 0.75
        args.warmup_times = 4
    elif args.config == 40:
        args.cache = 'prefix'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
        args.use_bd=True
        args.block_length=32
    elif args.config == 41:
        args.cache = 'prefix'
        args.parallel_decoding = 'threshold'
        args.prefix_look = 0
        args.after_look = 0
        args.threshold = 0.95
        args.warmup_times = 0
        args.use_bd=True
    procs = []
    print(args)

    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    args.tp_size = len(gpus)
    args.use_tp = args.tp_size > 1
    args.port_offset = gpus[0]
    
    if len(gpus) == 1:
        main(1, 0, gpus[0], args)
    else:
        for i, gpu in enumerate(gpus):
            p = Process(target=main, args=(len(gpus), i, gpu, args))
            p.daemon = True
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
