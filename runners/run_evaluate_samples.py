# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on SuperGLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""
import argparse
import json
import os
import shutil
import pickle as pkl
import time

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from utilities.utils import init_gpu_params, logger, set_seed
from utilities.grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from utilities.lm_seqs_dataset_modified import LmSeqsDataset
from distillation.evaluator import Evaluator
from transformers_local import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel
)




MODEL_CLASSES = {
    "gpt2": (
        GPT2Config,
        GPT2Tokenizer,
        GPT2LMHeadModel,
    ),
    "distilgpt2": (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    ),
}





def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--force", action="store_true", help="Overwrite output_dir if it already exists.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory (log, checkpoints, parameters, etc.)")
    parser.add_argument("--data_file", default=None, type=str, required=True, help="The input data path. Should contain the binarized file (or other data files) for the task.")

    parser.add_argument("--student_type", type=str, choices=["distilgpt2", "gpt2"], required=True, help="The student type (DistilGPT2, GPT2).")
    parser.add_argument("--student_config", type=str, required=True, help="Path to the student configuration.")
    parser.add_argument("--student_pretrained_weights", default=None, type=str, help="Load student initialization checkpoint.")
    parser.add_argument("--teacher_type", choices=["gpt2"], required=True, help="Teacher type (GPT2).")
    parser.add_argument("--teacher_name_or_path", type=str, required=True, help="The teacher type or weights.")
    
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce_student", default=1.0, type=float, help="Linear weight for the language modeling loss. Must be >=0.")
    parser.add_argument("--alpha_ce_teacher", default=1.0, type=float, help="Linear weight for the language modeling loss. Must be >=0.")
    parser.add_argument("--alpha_div", default=1.0, type=float, help="Linear weight for the kl divergence loss. Must be >=0.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (for each process).")
    parser.add_argument("--block_size", default=1024, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--seed", type=int, default=56, help="Random seed")

    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="Checkpoint interval.")
    parser.add_argument("--deepspeed", default=None, type=str, help="Path to deepspeed configuration.")
    parser.add_argument("--use_gpuid", type=int, default=-1, help="Use a specific GPU only")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--group_by_size", action="store_true", help="If true, group sequences that have similar length into the same batch. Default is true.")


    args = parser.parse_args()

    # We can now instantiate global lock
    init_gpu_params(args)
    # Set seed
    set_seed(args)

    if args.is_master:
        if os.path.exists(args.output_dir):
            if not args.force:
                raise ValueError(
                    f"Serialization dir {args.output_dir} already exists, but you have not precised wheter to overwrite it"
                    "Use `--force` if you want to overwrite it"
                )
            else:
                shutil.rmtree(args.output_dir)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info(f"Experiment will be dumped and logged in {args.output_dir}")

        # SAVE PARAMS #
        logger.info(f"Param: {args}")
        with open(os.path.join(args.output_dir, "parameters.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    ###########
    # Setup CUDA, GPU & distributed training
    if args.use_gpuid > -1:
        #device = args.use_gpuid
        args.n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        #device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    
    #args.device = device

     # TOKENIZER #
    student_config_class, _, student_model_class = MODEL_CLASSES[args.student_type]
    _, teacher_tokenizer_class, teacher_model_class = MODEL_CLASSES[args.teacher_type]
    tokenizer = teacher_tokenizer_class.from_pretrained(args.teacher_type)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    special_tok_ids = {}
    for tok_name, tok_symbol in tokenizer.special_tokens_map.items():
        idx = tokenizer.all_special_tokens.index(tok_symbol)
        special_tok_ids[tok_name] = tokenizer.all_special_ids[idx]
    logger.info(f"Special tokens {special_tok_ids}")
    special_tok_ids = special_tok_ids
    max_model_input_size = tokenizer.max_model_input_sizes[args.teacher_type] 

    # STUDENT #
    logger.info(f"Loading student config from {args.student_config}")
    stu_architecture_config = student_config_class.from_pretrained(args.student_config)
    stu_architecture_config.output_hidden_states = True

    if args.student_pretrained_weights is not None:
        logger.info(f"Loading pretrained weights from {args.student_pretrained_weights}")
        student = student_model_class.from_pretrained(args.student_pretrained_weights, config=stu_architecture_config)
    else:
        student = student_model_class(stu_architecture_config)

    if args.n_gpu > 0:
        student.to(f"cuda:{args.local_rank}")
    logger.info("Student loaded.")  

    # TEACHER #
    teacher = teacher_model_class.from_pretrained(args.teacher_name_or_path, output_hidden_states=True)
    if args.n_gpu > 0:
        teacher.to(f"cuda:{args.local_rank}")
    logger.info(f"Teacher loaded from {args.teacher_name_or_path}.")


    # DATA LOADER #
    logger.info(f"Loading data from file {args.data_file}")
    start = time.process_time()
    #train_lm_seq_dataset = LmSeqsDataset(params=args, max_model_input_size=max_model_input_size, special_tok_ids=special_tok_ids , protocol='pkl', data_dir=args.data_dir)
    with open(args.data_file, 'rb') as fp:
        data = pkl.load(fp)
    train_lm_seq_dataset = LmSeqsDataset(params=args, max_model_input_size=max_model_input_size, special_tok_ids=special_tok_ids , data=data)
    logger.info(f"Elapsed time for loading data = {time.process_time() - start} seconds...")   
    logger.info("Data loader created!")

    """
    if args.is_master:
        dir = '/media/data/yassir/datasets/openwebtext/owt_samples_data'
        logger.info(f"Saving now the data to {dir}")
        start = time.process_time()
        train_lm_seq_dataset.save_data(as_format = 'pkl', dump_dir = dir)
        logger.info(f"Elapsed time for saving data = {time.process_time() - start} seconds...") 
    """
    torch.cuda.empty_cache()
    evaluator = Evaluator(params=args, dataset=train_lm_seq_dataset, student=student, teacher=teacher)
    logger.info("Let's go get some drinks.")
    evaluator.evaluate_samples()
    logger.info("D-O-N-E")
    """
    logger.info(f"Loading data from {args.data_dir}")
    with open(args.data_dir, "rb") as fp:
        data = pkl.load(fp)
    output_dir = '/media/nvme/yassir/datasets/openwebtext/owt_samples_data'
    train_lm_seq_dataset = LmSeqsDataset(max_model_input_size=max_model_input_size, special_tok_ids=special_tok_ids , data=data)
    
    logger.info(f"Saving data in folder {args.output_dir} as PKL")
    start = time.process_time()
    train_lm_seq_dataset.save_data(as_format = 'pkl', dump_dir = args.output_dir)
    logger.info(f"Elapsed time = {time.process_time() - start} seconds...")
    logger.info(f"Saving data in folder {args.output_dir} as Tensors")
    start = time.process_time()
    train_lm_seq_dataset.save_data(as_format = 'ts', dump_dir = args.output_dir)
    logger.info(f"Elapsed time = {time.process_time() - start} seconds...")
    logger.info(f"Saving data in folder {args.output_dir} as NP arrays")
    start = time.process_time()
    train_lm_seq_dataset.save_data(as_format = 'np', dump_dir = args.output_dir)
    logger.info(f"Elapsed time = {time.process_time() - start} seconds...")   
    logger.info("D-O-N-E")

    start = time.process_time()
    logger.info(f"Saving data in folder {args.output_dir} as pkl")
    train_lm_seq_dataset.save_data(as_format = 'pkl', dump_dir = args.output_dir)
    logger.info(f"Elapsed time = {time.process_time() - start} seconds...")   

    
    logger.info(f"Loading data from folder {args.output_dir}")
    start = time.process_time()
    train_lm_seq_dataset = LmSeqsDataset(max_model_input_size=max_model_input_size, special_tok_ids=special_tok_ids , protocol='pkl', data_dir='/media/nvme/yassir/datasets/openwebtext/owt_samples_data')
    logger.info(f"Elapsed time = {time.process_time() - start} seconds...")   
    logger.info("D-O-N-E")
    """


if __name__ == "__main__":
    main()
