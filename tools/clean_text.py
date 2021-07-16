# coding=utf-8
"""
Generating samples from a given dataset
"""
import argparse
import numpy as np
import pickle as pkl
import json
import logging
from bs4 import BeautifulSoup
import warnings
import hashlib
import re
import os

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

logger = logging.getLogger(__name__)

def load_dataset(input_file):
    logger.info(f'-- Opening the file {input_file} ...')
    with open(input_file, "r", encoding="utf8") as fp:
        data = [line.rstrip() for line in fp]
        logger.info(f'-- Read {len(data)} lines')
    return data

def hash_sentence(sentence):
    no_punc_words = re.sub("[^\w\s]", "", sentence)
    words = no_punc_words.split()
    first, last = words[:4], words[-4:]
    first, last = ''.join(w for w in first), ''.join(w for w in last)
    return (str(len(no_punc_words)), hashlib.md5((first+last).encode()).hexdigest())

def save_records(args, dupps, html):
    logger.info(f'-- Saving data...')
    if len(dupps) > 0:
        with open(os.path.join(args.output_dir, 'dupps.json'), 'w') as fp:
            json.dump(dupps, fp)
    if len(html) > 0:
        with open(os.path.join(args.output_dir, 'html.json'), 'w') as fp:
            json.dump(html, fp)
    logger.info(f'-- Data saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cleaning the dataset."
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--output_dir", default="./", type=str)
    args = parser.parse_args()
    lines = load_dataset(args.input_file)

    reference, dupps, html = {}, [], {}
    with open(args.output_file, 'w') as fp:
        for i, line in enumerate(lines):
            if ((i + 1) < len(lines)) and (((i + 1) % 1000000) == 0):
                logger.info(f'Processed {i + 1} lines.')
                save_records(args, dupps, html)
            key = hash_sentence(line)
            if key in reference:
                dupps.append(line)
                continue
            else:
                reference[key] = {i: line}
            try:
                soup = BeautifulSoup(line, "html.parser")
                if  bool(soup.find()) == True:
                    html[i] = line
                    continue
            except Exception as e:
                html[i] = line
                continue
            fp.write(line + '\n')
    
    save_records(args, dupps, html)
    logger.info(f'-- DONE')

