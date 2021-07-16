# coding=utf-8

"""
Preprocessing script before training DistilGPT2.
Specific to GPT2 -> DistilGPT2.
"""
import argparse

import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Config
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction some layers of the full GPT2LMHeadModel for Transfer Learned Distillation"
    )
    parser.add_argument("--model_path",  type=str)
    parser.add_argument("--config_path",  type=str)
    args = parser.parse_args()
    if args.model_path:
        model = GPT2LMHeadModel.from_pretrained(args.model_path)
    elif args.config_path:
        config = GPT2Config.from_json_file(args.config_path)
        model = GPT2LMHeadModel(config)
    else:
        raise ValueError('You should either provide a path to the full model "model_path" or at least a path to a configuration file "config_path".')

    ## Then we iterate on others
    state_dict = model.state_dict()
    n_layers = int((len(state_dict) - 5) / 14)
    
    print(n_layers, ' layers.' )
    print('Trainable parameters: ', model.num_parameters(only_trainable = True))
    """
    for k, v in state_dict.items():
        print('-> ', k, ' : (', type(v), ', ', v.shape ,')')
    
    conv  = torch.nn.AvgPool1d(kernel_size = 257, stride = 1, padding = 0, ceil_mode=False, count_include_pad=True)
    t = state_dict['transformer.wte.weight']
    t = state_dict['transformer.wpe.weight']
    t = state_dict['transformer.h.0.attn.c_attn.weight']

    t = state_dict["transformer.h.0.ln_1.weight"]

    print(conv(t[None, None, :]).squeeze().shape)
   """

