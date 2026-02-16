# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

parallel_decoding='threshold' # or hierarchy
length=2048 # generate length
block_length=32 # block length
model_path='/root/inclusion_data/separate_expert_model/LLaDA2.0-mini-preview' # your model path  
threshold=0.98 # threshold for parallel decoding
low_threshold=0.62 # low threshold for parallel decoding when using hierarchy mechanism
cache='prefix' # or 'prefix' for prefix cache; or '' if you don't want to use cache
warmup_times=0 # warmup times for cache
prefix_look=0
after_look=0
cont_weight=0 # cont weight
use_credit=False # enable credit for threshold mechanism
use_compile=True # use compile
tp_size=4 # tensor parallel size
gpus='0;1;2;3' # gpus for tensor parallel inference
parallel='tp' # 'tp' for tensor parallel or 'dp' for data parallel
output_dir='./outputs' # your customer output path
model_type='llada2' # llada2 (for llada2-mini) 
use_bd=True # use block diffusion
master_port="23458"
save_samples=False # save samples
enable_remask=True # enable remasking for threshold decoder
# for llada 1.5 use tasks gsm8k_llada1.5 mbpp_sanitized_llada1.5
# for llada2_mini use tasks gsm8k_llada_mini mbpp_sanitized_llada_mini
if [ parallel=='tp' ]; then
  for task in mbpp_sanitized_llada_mini; do
    output_path=${output_dir}/${task}
    python eval_dinfer_sglang.py --tasks ${task} \
    --confirm_run_unsafe_code --model dInfer_eval \
    --model_args model_path=${model_path},gen_length=${length},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},show_speed=True,save_dir=${output_path},parallel_decoding=${parallel_decoding},cache=${cache},warmup_times=${warmup_times},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},cont_weight=${cont_weight},use_credit=${use_credit},prefix_look=${prefix_look},after_look=${after_look},gpus=${gpus},model_type=${model_type},use_bd=${use_bd},master_port=${master_port},save_samples=${save_samples},enable_remask=${enable_remask} \
    --output_path ${output_path} --include_path ./tasks --apply_chat_template
  done
else
  echo "parallel must be tp"
fi
