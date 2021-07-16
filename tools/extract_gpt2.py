# coding=utf-8

"""
Preprocessing script before training DistilGPT2.
Specific to GPT2 -> DistilGPT2.
"""
import argparse

import torch

from transformers import (
    GPT2LMHeadModel
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction some layers of the full GPT2LMHeadModel for Transfer Learned Distillation"
    )
    parser.add_argument("--model_type", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dump_checkpoint", default="./new_model.bin", type=str)
    parser.add_argument("--strategy", default="distil", choices=["3l", "distil", "alt", "6b", "3b3t", "6t"])
    args = parser.parse_args()

    if args.model_path:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    elif args.model_type:
        model = GPT2LMHeadModel.from_pretrained(args.model_type)
    else:
        raise ValueError('"model_path" should contain a valid path to the model / model_type should be "gpt2".')

    state_dict = model.state_dict()
    compressed_sd = {}
    ## We copy the five layers first
    compressed_sd["lm_head.weight"] = state_dict["lm_head.weight"]
    compressed_sd["transformer.ln_f.bias"] = state_dict["transformer.ln_f.bias"]
    compressed_sd["transformer.ln_f.weight"] = state_dict["transformer.ln_f.weight"]
    compressed_sd["transformer.wpe.weight"] = state_dict["transformer.wpe.weight"]
    compressed_sd["transformer.wte.weight"] = state_dict["transformer.wte.weight"]
    ## Then we iterate on others
    n_layers = int((len(state_dict) - 5) / 14)
    layers = []
    if args.strategy == '3l':
        step = int(0.5 * n_layers / 6)
        layers = [0, int(n_layers / 2), n_layers - 1]
    elif args.strategy == '6t':
        step = int(0.5 * n_layers / 6)
        layers = list(range(int((n_layers * 0.5) + step - 1), n_layers, step))
    elif args.strategy == '6b':
        step = int(0.5 * n_layers / 6)
        layers = list(range(0, int(n_layers * 0.5) - step + 1, step))
    elif args.strategy == '3b3t':
        #step = int(0.5 * n_layers / 6)
        #layers = list(range(0, int(n_layers * 0.5) - (3 * step), step))
        #layers += [*range(int((n_layers * 0.5) + (4 * step) - 1), n_layers, step)]
        step = int(n_layers / 6)
        layers = list(range(0, int(n_layers * 0.5), step))
        print('first batch ', layers)
        layers += [*range(int(n_layers * 0.5) + step - 1, n_layers, step)]
        print(step, layers)
    elif args.strategy == 'alt':
        step = int(n_layers / 6)
        layers = list(range(0, n_layers, step))
    else:
        step = int(n_layers / 6)
        layers = list(range(0, 3 * step, step))
        layers += [*range(n_layers - 3 * step + 1, n_layers, step)]
    print('Will copy layers : { ', layers, ' }')
    for index_c, index in enumerate(layers):
        print(f"Layer {index_c} in compressed, layer {index} in bigger model.")
        compressed_sd[f"transformer.h.{index_c}.attn.bias"] = state_dict[f"transformer.h.{index}.attn.bias"]
        compressed_sd[f"transformer.h.{index_c}.attn.c_attn.bias"] = state_dict[f"transformer.h.{index}.attn.c_attn.bias"] 
        compressed_sd[f"transformer.h.{index_c}.attn.c_attn.weight"] = state_dict[f"transformer.h.{index}.attn.c_attn.weight"]
        compressed_sd[f"transformer.h.{index_c}.attn.c_proj.bias"] = state_dict[f"transformer.h.{index}.attn.c_proj.bias"]
        compressed_sd[f"transformer.h.{index_c}.attn.c_proj.weight"] = state_dict[f"transformer.h.{index}.attn.c_proj.weight"]
        #compressed_sd[f"transformer.h.{index_c}.attn.masked_bias"] = state_dict[f"transformer.h.{index}.attn.masked_bias"]
        compressed_sd[f"transformer.h.{index_c}.ln_1.bias"] = state_dict[f"transformer.h.{index}.ln_1.bias"]
        compressed_sd[f"transformer.h.{index_c}.ln_1.weight"] = state_dict[f"transformer.h.{index}.ln_1.weight"]
        compressed_sd[f"transformer.h.{index_c}.ln_2.bias"] = state_dict[f"transformer.h.{index}.ln_2.bias"]
        compressed_sd[f"transformer.h.{index_c}.ln_2.weight"] = state_dict[f"transformer.h.{index}.ln_2.weight"]
        compressed_sd[f"transformer.h.{index_c}.mlp.c_fc.bias"] = state_dict[f"transformer.h.{index}.mlp.c_fc.bias"]
        compressed_sd[f"transformer.h.{index_c}.mlp.c_fc.weight"] = state_dict[f"transformer.h.{index}.mlp.c_fc.weight"]
        compressed_sd[f"transformer.h.{index_c}.mlp.c_proj.bias"] = state_dict[f"transformer.h.{index}.mlp.c_proj.bias"]
        compressed_sd[f"transformer.h.{index_c}.mlp.c_proj.weight"] = state_dict[f"transformer.h.{index}.mlp.c_proj.weight"]
    
    print(f"Reference model's size: {len(state_dict)} layers ({n_layers} effective).")
    print(f"Compressed model's size: {len(compressed_sd)} layers.")
    print(f"Save transferred checkpoint to {args.dump_checkpoint}.")
    torch.save(compressed_sd, args.dump_checkpoint)
    '''
    state_dict = model.state_dict()
    compressed_sd = {}
    ## Then we iterate on others
    n_layers = int((len(state_dict) - 5) / 14)
    '''
    #print(n_layers, ' layers.' )
    #print('Trainable parameters: ', compressed_sd.num_parameters(only_trainable = True))
