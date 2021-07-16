# coding=utf-8

"""
Preprocessing script before training truncated GPT2 architectures.
"""
import argparse
from typing import List

import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Config
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Building truncated versions of GPT2."
    )
    parser.add_argument("--ref_model_type", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--ref_model_path", type=str)
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--dump_checkpoint", default="./new_model.bin", type=str)
    parser.add_argument('--layers','--list', nargs='+', required=True, type=int)
    args = parser.parse_args()

    print(f'Parsed layers {args.layers}')

    # Loading reference
    if args.ref_model_path:
        ref_model = GPT2LMHeadModel.from_pretrained(args.ref_model_path)
    elif args.ref_model_type:
        ref_model = GPT2LMHeadModel.from_pretrained(args.ref_model_type)
    else:
        raise ValueError('You should either provide a path to the full model "model_path" or at least a path to a configuration file "config_path".')
    ref_state_dict = ref_model.state_dict()
    ref_layers = int((len(ref_state_dict) - 5) / 14)
    print(f'Reference model size: {ref_layers} layers.')

    selected_layers = sorted(args.layers)
    if selected_layers[-1] > (ref_layers - 1):
        raise ValueError('The layers indexes are out of range.')

    # Loading new model
    config = GPT2Config.from_json_file(args.config_path)
    model = GPT2LMHeadModel(config)
    state_dict = model.state_dict()
    layers = int((len(state_dict) - 5) / 14)
    print(f'New model size: {layers} layers.')

    # Check 
    if len(selected_layers) != layers:
        raise ValueError('Number of layers to convolve should be equal to the model\'s depth.')

    # We process three specific layers
    ## 1 Embedding layers
    print('Processing embedding layers...')
    state_dict['transformer.wte.weight'] = ref_state_dict['transformer.wte.weight']
    state_dict['transformer.wpe.weight'] = ref_state_dict['transformer.wpe.weight']
    ## 2 Language Modeling layer
    print('Processing Language Modeling layer...')
    state_dict['lm_head.weight'] = ref_state_dict['lm_head.weight']
    ## 3 Linear layer
    print('Processing linear layer...')
    state_dict['transformer.ln_f.bias'] = ref_state_dict['transformer.ln_f.bias']
    state_dict['transformer.ln_f.weight'] = ref_state_dict['transformer.ln_f.weight']

    # Then the other layers 
    print('Processing remaining layers:')
    for i, l in enumerate(selected_layers):
        print(f'Copying teacher layer {l} into student layer {i} ...')
        state_dict[f'transformer.h.{i}.ln_1.weight'] = ref_state_dict[f'transformer.h.{l}.ln_1.weight']
        state_dict[f'transformer.h.{i}.ln_1.bias'] = ref_state_dict[f'transformer.h.{l}.ln_1.bias']
        state_dict[f'transformer.h.{i}.attn.c_attn.weight'] = ref_state_dict[f'transformer.h.{l}.attn.c_attn.weight']
        state_dict[f'transformer.h.{i}.attn.c_attn.bias'] = ref_state_dict[f'transformer.h.{l}.attn.c_attn.bias']
        state_dict[f"transformer.h.{i}.attn.masked_bias"] = ref_state_dict[f"transformer.h.{l}.attn.masked_bias"]
        state_dict[f'transformer.h.{i}.attn.c_proj.weight'] = ref_state_dict[f'transformer.h.{l}.attn.c_proj.weight']
        state_dict[f'transformer.h.{i}.attn.c_proj.bias'] = ref_state_dict[f'transformer.h.{l}.attn.c_proj.bias']
        state_dict[f'transformer.h.{i}.ln_2.weight'] = ref_state_dict[f'transformer.h.{l}.ln_2.weight']
        state_dict[f'transformer.h.{i}.ln_2.bias'] = ref_state_dict[f'transformer.h.{l}.ln_2.bias']
        state_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = ref_state_dict[f'transformer.h.{l}.mlp.c_fc.weight']
        state_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = ref_state_dict[f'transformer.h.{l}.mlp.c_fc.bias']
        state_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = ref_state_dict[f'transformer.h.{l}.mlp.c_proj.weight']
        state_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = ref_state_dict[f'transformer.h.{l}.mlp.c_proj.bias']
    
    print(f"Save transferred checkpoint to {args.dump_checkpoint}.")
    torch.save(state_dict, args.dump_checkpoint)



    
    
