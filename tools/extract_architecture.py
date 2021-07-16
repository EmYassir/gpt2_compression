import torch
import argparse
from torch import nn
#from torch import init
from transformers import(
    GPT2Config,
    GPT2LMHeadModel,
)
from transformers.modeling_utils import Conv1D

def init_weights(module, init_range=0.02):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, Conv1D)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=init_range)
        #init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=init_range)
        #init.xavier_uniform_(module.weight)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction some layers of the full GPT2LMHeadModel for Transfer Learned Distillation"
    )
    parser.add_argument("--model_type", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dump_checkpoint", default="./new_model.bin", type=str)
    parser.add_argument("--strategy", default="emb", choices=["emb", "emb_top", "top", "no_init"])
    args = parser.parse_args()

    # Loading parent
    print('Loading reference model...')
    if not args.model_path:
        reference = GPT2LMHeadModel.from_pretrained(args.model_type)
    else:
        reference = GPT2LMHeadModel.from_pretrained(args.model_path)
    ref_state_dict = reference.state_dict()
    
    print('Building new model...')
    # model weights
    config = GPT2Config.from_json_file(args.config_file)
    model = GPT2LMHeadModel(config)

    print('Reinitializing new model...')
    # Re-init model weights
    model.apply(init_weights)
    compressed_sd = model.state_dict()

    print('Copying layers to new model...')
    if args.strategy == 'emb_top':
        compressed_sd["lm_head.weight"] = ref_state_dict["lm_head.weight"]
        compressed_sd["transformer.wpe.weight"] = ref_state_dict["transformer.wpe.weight"]
        compressed_sd["transformer.wte.weight"] = ref_state_dict["transformer.wte.weight"]
    elif args.strategy == 'top':
        compressed_sd["lm_head.weight"] = ref_state_dict["lm_head.weight"]
    elif args.strategy == 'emb':
        compressed_sd["transformer.wpe.weight"] = ref_state_dict["transformer.wpe.weight"]
        compressed_sd["transformer.wte.weight"] = ref_state_dict["transformer.wte.weight"]  
    else:
        pass
    ## Save
    torch.save(compressed_sd, args.dump_checkpoint)
    print(f"Saved model to {args.dump_checkpoint}.")


