import sys
import os
import shutil
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd() 

sys.path.append(current_dir)

from modeling_fused_lladamoe import FusedLLaDAMoEModelLM
from transformers import AutoTokenizer, AutoModel

def convert_and_save(
    input_path: str,
    output_path: str,
    modeling_file_name: str,
    device: str = "cpu"
):
    """
    Converts a standard OlmoeForCausalLM model to a FusedOlmoeForCausalLM model
    by fusing the MoE expert weights.
    """
    print(f"Loading original model from {input_path}...")
    config = AutoConfig.from_pretrained(input_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(input_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    state_dict = model.state_dict()

    num_layers = config.num_hidden_layers
    num_experts = config.num_experts
    print(f"Model config found: {num_layers} layers, {num_experts} experts per layer.")


    print("Building fused model...")
    fused_model = FusedLLaDAMoEModelLM(config).to(device)
    fused_state_dict = fused_model.state_dict()

    print("Mapping and fusing expert weights...")
    for i in range(num_layers):
        layer_prefix = f"model.layers.{i}.mlp."

        gate_weights = [state_dict[f"{layer_prefix}experts.{j}.gate_proj.weight"].to(device) for j in range(num_experts)]
        up_weights = [state_dict[f"{layer_prefix}experts.{j}.up_proj.weight"].to(device) for j in range(num_experts)]
        down_weights = [state_dict[f"{layer_prefix}experts.{j}.down_proj.weight"].to(device) for j in range(num_experts)]

        combined_w1 = torch.stack([torch.cat([g, u], dim=0) for g, u in zip(gate_weights, up_weights)])
        combined_w2 = torch.stack(down_weights)

        fused_state_dict[f"{layer_prefix}w1"] = combined_w1
        fused_state_dict[f"{layer_prefix}w2"] = combined_w2
    
    print("Copying non-expert parameters...")
    for key in state_dict:
        if 'experts' not in key:
            fused_state_dict[key] = state_dict[key]

    fused_model.load_state_dict(fused_state_dict)

    print("Updating model configuration for fused model...")
    if not hasattr(fused_model.config, "auto_map"):
        fused_model.config.auto_map = {}

    fused_model_class_name = FusedLLaDAMoEModelLM.__name__

    full_module_class_path = f"{modeling_file_name}.{fused_model_class_name}"

    fused_model.config.auto_map["AutoModelForCausalLM"] = full_module_class_path
    fused_model.config.auto_map.pop("AutoModel")
    fused_model.config.architectures = [fused_model_class_name]
    fused_model.config.architectures = [fused_model_class_name]

    print(f"Saving fused model to {output_path}")
    fused_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(input_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("Copying custom modeling files to output directory...")
    source_files = [
        os.path.join(current_dir, f"transfer/{modeling_file_name}.py"),
        os.path.join(current_dir, "transfer/fuse_moe.py")
    ]

    os.makedirs(output_path, exist_ok=True)

    for src_file in source_files:
        if os.path.exists(src_file):
            dest_file = os.path.join(output_path, os.path.basename(src_file))
            try:
                shutil.copy2(src_file, dest_file) 
                print(f"Copied {os.path.basename(src_file)} to {output_path}")
            except Exception as e:
                print(f"Error copying {os.path.basename(src_file)}: {e}")
        else:
            print(f"Warning: Source file not found, skipping copy: {src_file}")


    print("✅ Conversion completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--modeling', type=str, default='modeling_fused_olmoe')
    args = parser.parse_args()

    input_path = args.input.rstrip('/')
    output_path = args.output

    print(f"\n----- Starting CPU conversion for {input_path} -----")
    convert_and_save(
        input_path=input_path,
        output_path=output_path,
        modeling_file_name=args.modeling,
    )
    print(f"----- Finished conversion for {input_path} -> {output_path} -----\n")