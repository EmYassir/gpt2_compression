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
import os
import argparse
import logging
import random
import time
import json
from tqdm import tqdm
import numpy as np
import pickle as pkl
from bs4 import BeautifulSoup
import warnings
import multiprocessing
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)

def test_nonalpha_percent(sentence):
    counter, size = 0, len(sentence)
    if size == 0:
        return 0
    for word in sentence.split():
        for c in word:
            if not c.isalpha():
                counter += 1
    return counter/size

def extract_text_from_html(soup):
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '. '.join(chunk for chunk in chunks if chunk)
    return text


def preprocess_sample(args, sample):
    text = sample
    if not text or len(text) < args.min_length:
        return None
    
    ## HTML
    if args.html:
        try:
            soup = BeautifulSoup(text, "html.parser")
            if  bool(soup.find()) == True:
                text = extract_text_from_html(soup)
        except Exception as e:
            return None
        if len(text) < args.min_length:
            return None
    
    ## Number of alphanumerical symbols
    if test_nonalpha_percent(text) > args.percent_nonalpha:
        return None
    return text

    
        

def main():
    parser = argparse.ArgumentParser(
        description="Gather some statistics about data."
    )
    parser.add_argument("--data_path", type=str, default="./data/data.txt", help="The path to the data.")
    parser.add_argument("--output_file", type=str, default="./data/output.txt", help="The dump directory.")
    parser.add_argument("--html", action="store_true", help="Detects html in samples.")
    parser.add_argument("--percent_nonalpha", type=float, default=0.10, help="Percentage of non alphanumerical characters tolerated.")
    parser.add_argument("--min_length", type=int, default=5,  help="Minimum number of words.")
    args = parser.parse_args()

    logger.info(f"Loading text from {args.data_path}")
    with open(args.data_path, "r", encoding="utf8") as fp:
        data = [line.rstrip() for line in fp]
    
    logger.info(f"Starting preprocessing...")
    """
    for i, sample in enumerate(data):
        text = sample
        if not sample or len(sample) < args.min_length:
            continue
        l = len(sample.split())
        if l in dico:
            dico[l].append(i)
        else:
            dico[l] = [i]
        if args.html:
            try:
                soup = BeautifulSoup(sample, "html.parser")
                if  bool(soup.find()) == True:
                    text = extract_text_from_html(soup)
            except Exception as e:
                bad[i] = sample
        if i % 10000 == 0:
            logger.info(f'Processed {i} samples...')
    """
    inputs = tqdm(data)
    num_cores = multiprocessing.cpu_count()
    logger.info(f'Preprocessing data using {num_cores} cpu cores...')
    processed_list = Parallel(n_jobs=num_cores)(delayed(preprocess_sample)(args, sample) for sample in inputs)
    logger.info(f'Obtained {len(processed_list)} samples!')
    logger.info(f'Writing the cleaned samples to {args.output_file}...')
    with open(args.output_file, "w", encoding="utf8") as fp:
        for element in processed_list:
            if element is None:
                continue
            fp.write(element + '\n')



if __name__ == "__main__":
    main()