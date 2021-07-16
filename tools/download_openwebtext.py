# coding=utf-8

"""
Preprocessing script before distillation.
"""
import argparse
from datasets import load_dataset
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Downloads openwebtext and saves it to the disk."
    )
    parser.add_argument("--output_file", type=str, default=None, help="The dump file.")
    args = parser.parse_args()
    logger.info(f"Downloading data ...")
    dataset = load_dataset("openwebtext", "plain_text", split="train", ignore_verifications=True)
    logger.info(f"Saving data to {args.output_file}...")
    text = dataset['text']
    with open(args.output_file, 'w') as fp:
        for line in text:
            if len(line) > 0:
                fp.write(line.rstrip() + '\n')
    logger.info(f"Done")


if __name__ == "__main__":
    main()