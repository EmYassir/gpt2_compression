import torch
import argparse
from transformers import(
    GPT2Config,
    GPT2LMHeadModel
)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extraction some layers of the full GPT2LMHeadModel for Transfer Learned Distillation"
    )
    #parser.add_argument("--model_type", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dump_checkpoint", default="./new_model.bin", type=str)
    parser.add_argument("--strategy", default="embedding_only", choices=["embedding_only", "embedding_top", "top_only", "uninitialized"])
    args = parser.parse_args()

    # Loading parent
    parent = GPT2LMHeadModel.from_pretrained(args.model_path)
    
    # Loading child
    config = GPT2Config.from_json_file(args.config_file)
    child = GPT2LMHeadModel(config)

    state_dict = parent.state_dict()
    compressed_sd = {}
    
    if args.strategy == 'embedding_top':
        compressed_sd["lm_head.weight"] = state_dict["lm_head.weight"]
        compressed_sd["transformer.wpe.weight"] = state_dict["transformer.wpe.weight"]
        compressed_sd["transformer.wte.weight"] = state_dict["transformer.wte.weight"]
    elif args.strategy == 'top_only':
        compressed_sd["lm_head.weight"] = state_dict["lm_head.weight"]
    elif args.strategy == 'uninitialized':
        pass   
    else:
        compressed_sd["transformer.wpe.weight"] = state_dict["transformer.wpe.weight"]
        compressed_sd["transformer.wte.weight"] = state_dict["transformer.wte.weight"]

    ## Save
    n_layers = int((len(state_dict) - 5) / 14)    
    print(f"Reference model's size: {len(state_dict)} layers ({n_layers} effective).")
    print(f"Compressed model's size: {len(compressed_sd)} layers.")
    print(f"Trainable parameters: {compressed_sd.num_parameters(only_trainable = True)}")
    print(f"Saving transferred checkpoint to {args.dump_checkpoint}.")
    torch.save(compressed_sd, args.dump_checkpoint)


