import os
import logging
from multiprocessing import Process
import random

import torch
import torch.distributed as dist

from dinfer.decoding.utils import TokenArray, DistAlignedTokenArray
from dinfer.decoding.utils import TokenArray, BlockIterator, gather_sequence_block
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def test_block_iterator():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    x = TokenArray(prompt, gen_length=10, mask_id=17, eos_id=18, device='cpu')
    it = BlockIterator(x, block_length=5)
    num_iters = 0
    for block_id, (block_loc, block) in enumerate(it):
        num_iters += 1
        assert block_loc.start == block_id * 5 + prompt.shape[1]
        assert block_loc.end == min((block_id + 1) * 5 + prompt.shape[1], prompt.shape[1] + 10)
    assert num_iters == 2
  
def test_token_array():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    arr = TokenArray(prompt, gen_length=20, mask_id=17, eos_id=18, device='cpu')
    assert arr.total_length == prompt.shape[1] + 20
    assert torch.all(arr[:, 0:5] == prompt[:, 0:5])
    arr[:, 8:10] = torch.tensor([9, 10]).view(1, 2)

    arr = DistAlignedTokenArray(prompt, gen_length=20, mask_id=17, eos_id=18, device='cpu', rank=0, world_size=4)
    assert arr.total_length == prompt.shape[1] + 20 + 1
    assert torch.all(arr[:, 0:5] == prompt[:, 0:5])
    arr[:, 8:10] = torch.tensor([9, 10]).view(1, 2)


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '37865'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def check_worker(rank, world_size, gpu):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    full_data = torch.arange(100).view(4, 25).to(device)

    # Partition size is smaller than block size.
    block_size = 6
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is larger than block size.
    block_size = 4
    part_size = 6
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is equal to block size.
    block_size = 4
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    dist.destroy_process_group()


def test_dist():
    num_gpus = 4
    procs = []
    for i, gpu in enumerate(range(num_gpus)):
        p = Process(target=check_worker, args=(i, num_gpus, i))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO)

    test_block_iterator()
    print('test_block_iterator passed')
    test_token_array()
    print('test_token_array passed')
    test_dist()
    print('test_dist passed')