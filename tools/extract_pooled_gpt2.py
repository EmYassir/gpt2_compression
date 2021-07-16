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
        description="Building 'pooled' versions of GPT2."
    )
    parser.add_argument("--ref_model_type", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--ref_model_path", type=str)
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--dump_checkpoint", default="./new_model.bin", type=str)
    parser.add_argument("--pooled_size", required=True, type=int)
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
    ref_model = ref_model.to("cuda:0")
    ref_state_dict = ref_model.state_dict()
    ref_layers = int((len(ref_state_dict) - 5) / 14)
    print(f'Reference model size: {ref_layers} layers.')

    selected_layers = sorted(args.layers)
    if selected_layers[-1] > (ref_layers - 1):
        raise ValueError('The layers indexes are out of range.')

    # Loading new model
    config = GPT2Config.from_json_file(args.config_path)
    model = GPT2LMHeadModel(config)
    model = model.to("cuda:0")
    state_dict = model.state_dict()
    layers = int((len(state_dict) - 5) / 14)
    print(f'New model size: {layers} layers.')

    # Check 
    if len(selected_layers) != layers:
        raise ValueError('Number of layers to convolve should be equal to the model\'s depth.')

    # Start pooling
    ref_embed_size = ref_model.lm_head.in_features
    p = args.pooled_size

    # Create pooling layers
    print('Creating pooling layers...')
    conv1D1E  = torch.nn.AvgPool1d(kernel_size = ref_embed_size - p + 1, stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
    conv1D3E  = torch.nn.AvgPool1d(kernel_size = 3*(ref_embed_size - p) + 1, stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
    conv1D4E  = torch.nn.AvgPool1d(kernel_size = 4*(ref_embed_size - p) + 1, stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
    conv2D1E3E = torch.nn.AvgPool2d(kernel_size = (ref_embed_size - p + 1, 3*(ref_embed_size - p) + 1) , stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
    conv2D1E1E = torch.nn.AvgPool2d(kernel_size = (ref_embed_size - p + 1, ref_embed_size - p + 1) , stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
    conv2D1E4E = torch.nn.AvgPool2d(kernel_size = (ref_embed_size - p + 1, 4*(ref_embed_size - p) + 1) , stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
    conv2D4E1E = torch.nn.AvgPool2d(kernel_size = (4*(ref_embed_size - p) + 1, ref_embed_size - p + 1) , stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)

    # We process three specific layers
    ## 1 Embedding layers
    print('Processing embedding layers...')
    state_dict['transformer.wte.weight'] = conv1D1E(ref_state_dict['transformer.wte.weight'][None, :]).squeeze()
    print('Processed \'transformer.wte.weight\': output shape = ', state_dict['transformer.wte.weight'].shape)
    state_dict['transformer.wpe.weight'] = conv1D1E(ref_state_dict['transformer.wpe.weight'][None, :]).squeeze()
    print('Processed \'transformer.wpe.weight\': output shape = ', state_dict['transformer.wpe.weight'].shape)
    ## 2 Language Modeling layer
    print('Processing Language Modeling layer...')
    state_dict['lm_head.weight'] = conv1D1E(ref_state_dict['lm_head.weight'].unsqueeze(0)).squeeze()
    print('Processed \'lm_head.weight\': output shape = ', state_dict['lm_head.weight'].shape)
    ## 3 Linear layer
    print('Processing linear layer...')
    state_dict['transformer.ln_f.bias'] = conv1D1E(ref_state_dict['transformer.ln_f.bias'][None, None, :]).squeeze()
    print('Processed \'transformer.ln_f.bias\': output shape = ', state_dict['transformer.ln_f.bias'].shape)
    state_dict['transformer.ln_f.weight'] = conv1D1E(ref_state_dict['transformer.ln_f.weight'][None, None, :]).squeeze()
    print('Processed \'transformer.ln_f.weight\': output shape = ', state_dict['transformer.ln_f.weight'].shape)

    # Then the other layers 
    print('Processing remaining layers...')
    for i, l in enumerate(selected_layers):
        state_dict[f'transformer.h.{i}.ln_1.weight'] = conv1D1E(ref_state_dict[f'transformer.h.{l}.ln_1.weight'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.ln_1.weight\': output shape = ', state_dict[f'transformer.h.{i}.ln_1.weight'].shape)
        state_dict[f'transformer.h.{i}.ln_1.bias'] = conv1D1E(ref_state_dict[f'transformer.h.{l}.ln_1.bias'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.ln_1.bias\': output shape = ', state_dict[f'transformer.h.{i}.ln_1.bias'].shape)
        state_dict[f'transformer.h.{i}.attn.c_attn.weight'] = conv2D1E3E(ref_state_dict[f'transformer.h.{l}.attn.c_attn.weight'][None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.attn.c_attn.weight\': output shape = ', state_dict[f'transformer.h.{i}.attn.c_attn.weight'].shape)
        state_dict[f'transformer.h.{i}.attn.c_attn.bias'] = conv1D3E(ref_state_dict[f'transformer.h.{l}.attn.c_attn.bias'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.attn.c_attn.bias\': output shape = ', state_dict[f'transformer.h.{i}.attn.c_attn.bias'].shape)
        state_dict[f'transformer.h.{i}.attn.c_proj.weight'] = conv2D1E1E(ref_state_dict[f'transformer.h.{l}.attn.c_proj.weight'][None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.attn.c_proj.weight\': output shape = ', state_dict[f'transformer.h.{i}.attn.c_proj.weight'].shape)
        state_dict[f'transformer.h.{i}.attn.c_proj.bias'] = conv1D1E(ref_state_dict[f'transformer.h.{l}.attn.c_proj.bias'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.attn.c_proj.bias\': output shape = ', state_dict[f'transformer.h.{i}.attn.c_proj.bias'].shape)
        state_dict[f'transformer.h.{i}.ln_2.weight'] = conv1D1E(ref_state_dict[f'transformer.h.{l}.ln_2.weight'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.ln_2.weight\': output shape = ', state_dict[f'transformer.h.{i}.ln_2.weight'].shape)
        state_dict[f'transformer.h.{i}.ln_2.bias'] = conv1D1E(ref_state_dict[f'transformer.h.{l}.ln_2.bias'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.ln_2.bias\': output shape = ', state_dict[f'transformer.h.{i}.ln_2.bias'].shape)
        state_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = conv2D1E4E(ref_state_dict[f'transformer.h.{l}.mlp.c_fc.weight'][None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.mlp.c_fc.weight\': output shape = ', state_dict[f'transformer.h.{i}.mlp.c_fc.weight'].shape)
        state_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = conv1D4E(ref_state_dict[f'transformer.h.{l}.mlp.c_fc.bias'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.mlp.c_fc.bias\': output shape = ', state_dict[f'transformer.h.{i}.mlp.c_fc.bias'].shape)
        state_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = conv2D4E1E(ref_state_dict[f'transformer.h.{l}.mlp.c_proj.weight'][None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.mlp.c_proj.weight\': output shape = ', state_dict[f'transformer.h.{i}.mlp.c_proj.weight'].shape)
        state_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = conv1D1E(ref_state_dict[f'transformer.h.{l}.mlp.c_proj.bias'][None, None, :]).squeeze()
        print(f'Processed \'transformer.h.{i}.mlp.c_proj.bias\': output shape = ', state_dict[f'transformer.h.{i}.mlp.c_proj.bias'].shape)
    
    print(f"Save transferred checkpoint to {args.dump_checkpoint}.")
    torch.save(state_dict, args.dump_checkpoint)



    
    
