# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3


length=1024 # generate length
block_length=64 # block length
model_path='your model path'  
cache='dual' # or 'prefix' for prefix cache; or '' if you don't want to use cache
prefix_look=16 # prefix look length for cache
after_look=16 # after look length for cache
warmup_times=4 # warmup times for cache
cont_weight=0.3 # cont weight 
use_credit=False # use credit for threshold mechanism
use_compile=True # use compile
gpus='0,1,2,3' # gpus for tensor parallel inference
parallel='tp' # 'tp' for tensor parallel or 'dp' for data parallel
output_path='your customer output path'
task=mbpp_sanitized_llada

# use threshold mechanism for  parallel decoding
parallel_decoding='threshold' # or hierarchy
threshold=0.8 # threshold for parallel decoding
python eval_dinfer.py --tasks ${task} \
  --confirm_run_unsafe_code --model dInfer_eval \
  --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},prefix_look=${prefix_look},after_look=${after_look},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit} \
  --output_path ${output_path} --include_path ./tasks --apply_chat_template 

# # use accelerate to enable multi-gpu data parallel inference
# parallel=dp
# accelerate launch eval_dinfer.py --tasks ${task} \
# --confirm_run_unsafe_code --model dInfer_eval \
# --model_args model_path=${model_path},gen_length=${length}block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},prefix_look=${prefix_look},after_look=${after_look},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit},gpus=${gpus} \
# --output_path ${output_path} --include_path ../tasks --apply_chat_template

# use hierarchy mechanism for parallel decoding 
parallel_decoding='hierarchy' 
threshold=0.92 # threshold for parallel decoding
low_threshold=0.62 # low threshold for parallel decoding when using hierarchy mechanism
python eval_dinfer.py --tasks ${task} \
--confirm_run_unsafe_code --model dInfer_eval \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},prefix_look=${prefix_look},after_look=${after_look},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},parallel=${parallel} \
--output_path ${output_path} --include_path ../tasks --apply_chat_template --log_samples