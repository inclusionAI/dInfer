# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

parallel_decoding='threshold' # or hierarchy
length=1024 # generate length
block_length=64 # block length
model_path=''  
threshold=0.76 # threshold for parallel decoding
low_threshold=0.62 # low threshold for parallel decoding when using hierarchy mechanism
cache='dual' # or 'prefix' for prefix cache; or '' if you don't want to use cache
warmup_times=4 # warmup times for cache
prefix_look=16
after_look=16
cont_weight=0.3 # cont weight
use_credit=False # enable credit for threshold mechanism
use_compile=True # use compile
tp_size=4 # tensor parallel size
gpus='0,1,2,3' # gpus for tensor parallel inference
parallel='tp' # 'tp' for tensor parallel or 'dp' for data parallel
output_path='./res' # your customer output path
# for llada 1.5 use tasks gsm8k_llada1.5 mbpp_sanitized_llada1.5
# for llada moe use tasks gsm8k_llada_moe mbpp_sanitized_llada_moe
for task in gsm8k_llada_moe mbpp_sanitized_llada_moe; do
  python eval_dinfer.py --tasks ${task} \
  --confirm_run_unsafe_code --model dInfer_eval \
  --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path}/${task},parallel_decoding=${parallel_decoding},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit},prefix_look=${prefix_look},after_look=${after_look}\
  --output_path ${output_path}/${task} --include_path ./tasks --apply_chat_template
done

# use accelerate to enable multi-gpu data parallel inference
# parallel=dp
# accelerate launch eval_dinfer.py --tasks ${task} \
# --confirm_run_unsafe_code --model dInfer_eval \
# --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},prefix_look=${prefix_look},after_look=${after_look},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit},gpus=${gpus} \
# --output_path ${output_path} --include_path ./tasks --apply_chat_template