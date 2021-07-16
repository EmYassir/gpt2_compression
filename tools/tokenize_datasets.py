import logging
import os
import sys
import pickle as pkl
import argparse
import traceback

from multiprocessing import cpu_count, get_context, Pool

from datasets import load_dataset, load_from_disk

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

tokenizer = None
block_size = 1024
logger = logging.getLogger(__name__)
text_column_name = None


def save_to_file(obj, file_path):
    with open(file_path, 'wb') as fp:
        pkl.dump(obj, fp, protocol=pkl.HIGHEST_PROTOCOL)


def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def parallel_compute(func, iterable, cores = None, chunk=1000):
    if not cores:
        cores = cpu_count() - 1
    with get_context('fork').Pool(cores) as p:
        results =  p.map(func, iterable, chunksize=chunk)
        p.close()
        p.join()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenizer"
    )
    parser.add_argument("--tokenizer_type_or_path", required=True, type=str)
    parser.add_argument("--block_size", default=block_size, type=str)
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--valid_data", type=str)
    parser.add_argument("--tokens_dir", required=True, type=str)
    parser.add_argument("--grouped_tokens_dir", required=True, type=str)
    parser.add_argument("--group_only", action="store_true")
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()
    
    if not os.path.isdir(args.tokens_dir):
        raise ValueError('"tokens_dir" should be a valid directory.')
    if not os.path.isdir(args.grouped_tokens_dir):
        raise ValueError('"grouped_tokens_dir" should be a valid directory.')
    
    if not args.group_only:
        if not os.path.isfile(args.train_data):
            raise ValueError('"train_data" should be a valid file path.')
        if (args.valid_data is not None) and (not os.path.isfile(args.valid_data)):
            raise ValueError('"valid_data" should be a valid file path.')

    tokenizer_kwargs = {"use_fast": True}
    print("Loading pre-trained tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_type_or_path, **tokenizer_kwargs)
    except Exception as e:
        logging.error(traceback.format_exc())
        raise ValueError('"tokenizer_type_or_path" should be a valid type/directory.')

    if not args.group_only:
        print("********* Loading datafiles")
        data_files = {}
        data_files["train"] = args.train_data
        if args.valid_data is not None:
            data_files["validation"] = args.valid_data
        extension = args.train_data.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files, ignore_verifications=True)

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        print(">>>>>>>>> [START] Tokenizing datasets ...")
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.num_proc,
            remove_columns=column_names,
            load_from_cache_file=True
        )
        print(">>>>>>>>> [END] Tokenizing datasets ...")
        print(f"Saving tokenized datasets to the directory \'{args.tokens_dir}\' ...")
        tokenized_datasets.save_to_disk(args.tokens_dir)
    else:
        print(">>>>>>>>> [START] Loading tokenized datasets ...")
        tokenized_datasets = load_from_disk(args.tokens_dir)
        print(">>>>>>>>> [END] Loading tokenized datasets ...")

    print(">>>>>>>>> [START] Grouping tokenized datasets ...")
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_proc,
        load_from_cache_file=True,
    )
    print(">>>>>>>>> [END] Grouping tokenized datasets ...")
    print(f"Saving grouped grouped-tokenized datasets to \'{args.grouped_tokens_dir}\' ...")
    lm_datasets.save_to_disk(args.grouped_tokens_dir)
    print("DONE")

    
