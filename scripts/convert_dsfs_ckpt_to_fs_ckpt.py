"""
Author: Tianyu Gao
Email: tianyug@cs.princeton.edu

Usage: 

python convert_dsfs_ckpt_to_fs_ckpt.py --fr {deepspeed fairseq checkpoint} --to {Fairseq checkpoint}
"""
import torch
import argparse

def convert_dsfs_ckpt_to_fs_ckpt(fr, to):
    print(f"Convert deepspeed fairseq checkpoint {fr} to fairseq checkpoint {to}")
    old_ckpt = torch.load(fr)
    new_ckpt = {"cfg": old_ckpt["cfg"]}
    for key in old_ckpt:
        if key not in ['cfg', 'model']:
            new_ckpt[key] = old_ckpt[key]

    # Turn off all deepspeed flags
    new_ckpt["cfg"]['model'].deepspeed_cuda_kernel = False
    new_ckpt["cfg"]['common']['deepspeed_cuda_kernel'] = False
    new_ckpt["cfg"]['task'].deepspeed_cuda_kernel = False

    if hasattr(old_ckpt["cfg"]['model'], 'arch'):
        new_ckpt["cfg"]['model'].arch = old_ckpt["cfg"]['model'].arch.split('deepspeed_', 1)[-1]
    if hasattr(old_ckpt["cfg"]['model'], '_name'):
        new_ckpt["cfg"]['model']._name = old_ckpt["cfg"]['model']._name.split('deepspeed_', 1)[-1]
    new_ckpt["cfg"]['model'].deepspeed_stochastic_mode = False
    new_ckpt["cfg"]['common']['deepspeed_stochastic_mode'] = False
    new_ckpt["cfg"]['task'].deepspeed_stochastic_mode = False

    new_ckpt["model"] = {}

    for key, param in old_ckpt["model"].items():
        if "attn_qkvw" in key:
            hs = param.size(0) // 3
            key_q = key.replace("attn_qkvw", "self_attn.q_proj.weight")
            key_k = key.replace("attn_qkvw", "self_attn.k_proj.weight")
            key_v = key.replace("attn_qkvw", "self_attn.v_proj.weight")
            new_ckpt["model"][key_q] = param[:hs]
            new_ckpt["model"][key_k] = param[hs:hs*2]
            new_ckpt["model"][key_v] = param[hs*2:]     
        elif "attn_qkvb" in key:
            hs = param.size(0) // 3
            key_q = key.replace("attn_qkvb", "self_attn.q_proj.bias")
            key_k = key.replace("attn_qkvb", "self_attn.k_proj.bias")
            key_v = key.replace("attn_qkvb", "self_attn.v_proj.bias")
            new_ckpt["model"][key_q] = param[:hs]
            new_ckpt["model"][key_k] = param[hs:hs*2]
            new_ckpt["model"][key_v] = param[hs*2:]
        else: 
            if "attn_ow" in key:
                new_key = key.replace("attn_ow", "self_attn.out_proj.weight")
            elif "attn_ob" in key:
                new_key = key.replace("attn_ob", "self_attn.out_proj.bias")
            elif "norm_w" in key:
                # Note that the deepspeed norm naming is weird
                # norm --> self attention layer norm
                # attn_n --> final layer norm
                new_key = key.replace("norm_w", "self_attn_layer_norm.weight") 
            elif "norm_b" in key:
                new_key = key.replace("norm_b", "self_attn_layer_norm.bias") 
            elif "inter_w" in key:
                new_key = key.replace("inter_w", "fc1.weight") 
            elif "inter_b" in key: 
                new_key = key.replace("inter_b", "fc1.bias") 
            elif "output_w" in key:
                new_key = key.replace("output_w", "fc2.weight") 
            elif "output_b" in key:
                new_key = key.replace("output_b", "fc2.bias") 
            elif "attn_nw" in key:
                new_key = key.replace("attn_nw", "final_layer_norm.weight") 
            elif "attn_nb" in key:
                new_key = key.replace("attn_nb", "final_layer_norm.bias") 
            else:
                new_key = key

            new_ckpt["model"][new_key] = param

    torch.save(new_ckpt, to)
    print(f"Saved fairseq checkpoint to {to}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fr", type=str)
    parser.add_argument("--to", type=str)
    args = parser.parse_args()
    convert_dsfs_ckpt_to_fs_ckpt(args.fr, args.to)
