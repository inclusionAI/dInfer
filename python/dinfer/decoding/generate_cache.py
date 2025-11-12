from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
import time

from .parellel_strategy import get_transfer_index_cache
from .utils import get_num_transfer_tokens

def cache_update_tag(strategy, **kwargs):
    """
        stratgry: determin update cache or not
    """
    if strategy == "fix_iter":
        # Fixed cache update per iteration
        if 'step' not in kwargs or 'iter' not in kwargs:
            raise ValueError("Missing required parameters 'step' and 'iter'")

        step = kwargs['step']
        iter = kwargs['iter']
        if not isinstance(step, int) or not isinstance(iter, int):
            raise TypeError("Both 'step' and 'iter' must be integers")

        if step % iter == 0:
            return True
        else:
            return False
    
def block_cache_api(model, x, past_key_values, position, block_mask_index, update_cache=False):
    """
    Input:
         model: llada model
         x:[1, prompt_len + gen_len] model input 
         past_key_value: past key values of model for x
         position [1, prompt+gen_len]: position to replace the KV cache with calculated KV,
         block_mask_index [1, prompt+gen_len]: positions of currect decoded blck.
         update_cache: bool type. Update cache or not. Use the cache_update_tag to determine this tag.
    Output:
    past
        past_key_values: past cache for all token, which may not efficient for prompt/prefix cache type.
        logits [1,block_len,|V|]

    """

    # generate kv cache
    if past_key_values is None or update_cache:
        # re calculate
        output = model(x, use_cache=True)        
        past_key_values = output.past_key_values
        logits = output.logits[:, block_mask_index[0]] #[:, b_start:b_end, |V|]


    else:
        # use KV cache to obatin logits
        new_mask = (position & block_mask_index)[:, position[0]] #[1, C]
        output = model(x[:, position[0]], past_key_values=past_key_values, use_cache=True, replace_position=position) # [1, C, V]
        logits = output.logits[:,new_mask[0]]
        
    return past_key_values, logits



@torch.no_grad()
def generate_with_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                mask_id=126336, log_flops=False, threshold=None, cache_update_iter=None, eos_early_stop=False, minimal_topk=1, **kwargs):

    '''
    force update cache in some iters.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        mask_id: The toke id of [MASK] is 126336.
    '''
    cache_type = kwargs.get("cache_type", "prefix") #[prefix, dual, prompt]
    log_flops = kwargs.get("log_flops", False)
    threshold = kwargs.get("threshold", None)
    cache_update_iter = kwargs.get("cache_update_iter", None)
    eos_early_stop = kwargs.get("eos_early_stop", False)
    minimal_topk = kwargs.get("minimal_topk", 1)



    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    start0 = time.time()

    seq_op_num = 0
    nfe = 0
    cache_update_step=[]
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block+1) * block_length
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache range, replace the kvcache with re-calculated score.
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        if cache_type=="dual":
            print("use dual cache")
            replace_position[:, current_block_start:current_block_end] = 1
        elif cache_type=="prefix":
            print("use prefix cache")
            replace_position[:, current_block_start:] = 1
        elif cache_type=="prompt":
            print("cache prompt")
            replace_position[:, prompt.shape[1]:] = 1

        block_mask = torch.zeros_like(x, dtype=torch.bool)
        block_mask[:, current_block_start:current_block_end] = 1


        i = 0
        past_key_values = None

        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            if log_flops:
                if cfg_scale > 0.:
                    cfg_factor = 2
                else:
                    cfg_factor = 1
                actual_shape = x[:, current_block_start:].shape
                op_num = cfg_factor * (32*(4*actual_shape[0]*4096*4096*actual_shape[1]*2 + actual_shape[0]*actual_shape[1]*actual_shape[1]*4096*2+
                        3*actual_shape[0]*4096*12288*actual_shape[1]*2) + actual_shape[0]*4096*126464*actual_shape[1]*2)/ 1e12 
                seq_op_num += op_num        
        
            
            mask_index = (x[:,current_block_start:current_block_end] == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if cache_update_iter is not None and i%cache_update_iter == 0:
                    # re-calculate kvcache
                    output = model(x_, use_cache=True)
                    logits = output.logits
                    past_key_values = output.past_key_values

                    # update kvcache
                    new_past_key_values = []
                    for k in range(len(past_key_values)):
                        new_past_key_values.append(())
                        for l in range(len(past_key_values[k])):
                            new_past_key_values[k] += (past_key_values[k][l][:, :, :prompt.shape[1] + num_block * block_length],)
                    past_key_values = new_past_key_values

                else:
                    # use kvcache directly
                    logits = model(x_[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                update_tag = cache_update_tag("fix_iter",step=i, iter=cache_update_iter)
                past_key_values, logits = block_cache_api(model, x, past_key_values, replace_position, block_mask, update_cache=update_tag)
                if update_tag:
                    cache_update_step.append(nfe)


            x0, transfer_index = get_transfer_index_cache(logits, mask_index, x, block_length, num_transfer_tokens[:, i], temperature, remasking="low_confidence",
                                                    threshold=threshold, minimal_topk=minimal_topk)

            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]

            i+=1
            # early stop: with eos_early_stop tag & + eos token appeared + this block has been unmasked
            if eos_early_stop and (x0[transfer_index] == 126081).any() and not (x[:, current_block_start:current_block_end]==mask_id).any():
                if log_flops:
                    end0 = time.time()
                    print('====sequence flops:', seq_op_num / (end0-start0), 'TFLOPs')
                    print("cache updated as step:", cache_update_step)

                # x[:,current_block_end:] = eos_id
                return x, nfe

    if log_flops:
        end0 = time.time()
        print('====sequence flops:', seq_op_num / (end0-start0), 'TFLOPs')
        print("cache updated as step:", cache_update_step)
    return x, nfe




@torch.no_grad()
def generate_with_prefixcache_update(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                mask_id=126336, log_flops=False, threshold=None, cache_update_iter=None, eos_early_stop=False, minimal_topk=1, **kwargs):

    '''
    force update cache in some iters.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        mask_id: The toke id of [MASK] is 126336.
    '''
    log_flops = kwargs.get("log_flops", False)
    threshold = kwargs.get("threshold", None)
    cache_update_iter = kwargs.get("cache_update_iter", None)
    eos_early_stop = kwargs.get("eos_early_stop", False)
    minimal_topk = kwargs.get("minimal_topk", 1)



    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    start0 = time.time()

    seq_op_num = 0
    nfe = 0
    cache_update_step=[]
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block+1) * block_length
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        i = 0
        past_key_values = None

        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            if log_flops:
                cfg_factor = 1
                actual_shape = x[:, current_block_start:].shape
                op_num = cfg_factor * (32*(4*actual_shape[0]*4096*4096*actual_shape[1]*2 + actual_shape[0]*actual_shape[1]*actual_shape[1]*4096*2+
                        3*actual_shape[0]*4096*12288*actual_shape[1]*2) + actual_shape[0]*4096*126464*actual_shape[1]*2)/ 1e12 
                seq_op_num += op_num        
        
            
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, current_block_end:] = False

            if cache_update_iter is not None and i%cache_update_iter == 0:
                cache_update_step.append(nfe)

                # re-calculate kvcache
                output = model(x, use_cache=True)
                logits = output.logits[:,current_block_start:]
                past_key_values = output.past_key_values

                # update kvcache
                new_past_key_values = []
                for k in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for l in range(len(past_key_values[k])):
                        new_past_key_values[k] += (past_key_values[k][l][:, :, :prompt.shape[1] + num_block * block_length],)
                past_key_values = new_past_key_values

            else:
                logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            x0, transfer_index = get_transfer_index_cache(logits, mask_index, x[:,current_block_start:], block_length, num_transfer_tokens[:, i], temperature, remasking,
                                                    threshold=threshold, minimal_topk=minimal_topk)

            x[:, current_block_start:][transfer_index] = x0[transfer_index]

            i+=1
            # early stop: with eos_early_stop tag & + eos token appeared + this block has been unmasked
            if eos_early_stop and (x0[transfer_index] == 126081).any() and not (x[:, current_block_start:current_block_end]==mask_id).any():
                if log_flops:
                    end0 = time.time()
                    print('====sequence flops:', seq_op_num / (end0-start0), 'TFLOPs')
                    print("cache updated as step:", cache_update_step)

                # x[:,current_block_end:] = eos_id
                return x, nfe

    if log_flops:
        end0 = time.time()
        print('====sequence flops:', seq_op_num / (end0-start0), 'TFLOPs')
        print("cache updated as step:", cache_update_step)
    return x, nfe








