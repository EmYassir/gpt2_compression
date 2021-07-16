# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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
"""
Preprocessing script before distillation.
"""
import argparse
import logging
from typing import List
import numpy as np
import pickle as pkl
from numpy.lib.format import open_memmap

from transformers import BertTokenizer, GPT2Tokenizer, RobertaTokenizer


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def dump(args):
    logger.info(f"Loading data from {args.file_path}")
    with open(args.file_path, "rb") as fp:
        data = pkl.load(fp)
    logger.info(">>>> Converting data to np array...")
    arr = np.array(data, dtype=object)
    logger.info(f">>>> Writing data to file {args.output_file}...")
    np.save(args.output_file, arr)




def load(args):
    logger.info(f">>>> Loading data from file {args.file_path}...")
    data = np.load(args.file_path, allow_pickle=True)
    #data = open_memmap(args.file_path, mode='r+', dtype=List, shape=(332011430,))
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids)."
    )
    parser.add_argument("--file_path", type=str, default="data/dump.pkl", help="The path to the data.")
    parser.add_argument("--output_file", type=str, default=None, help="The dump file prefix.")
    parser.add_argument("--length", type=int, default=332011430, help="The total length.")
    parser.add_argument("--flag", type=str, choices=['r', 'r+', 'c', 'w', 'w+'], default='r+', help="The mode.")
    args = parser.parse_args()
    if args.output_file is None:
        _ = load(args)
    else:
        dump(args)
    logger.info(f"Done")


if __name__ == "__main__":
    main()