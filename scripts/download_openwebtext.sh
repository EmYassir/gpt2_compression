#!/bin/sh

## CACHE
#export TRANSFORMERS_CACHE=/media/nvme/yassir/huggingface/
#export HF_HOME=/media/nvme/yassir/huggingface/
export TRANSFORMERS_CACHE=/d/.cache/huggingface/
export HF_HOME=/d/.cache/huggingface/


## MODELS
#export RUNNER=/home/yassir/gpt2-kd/tools/download_openwebtext.py
#export OUTPUT_FILE=/media/nvme/yassir/datasets/openwebtext/plain_text.txt
export RUNNER=/d/Projects/internship/gpt2-kd/tools/download_openwebtext.py
export OUTPUT_FILE=/d/Projects/internship/datasets/openwebtext/plain_text.txt

## START
echo ">>>>>> Start downloading OPENWEBTEXT";
python $RUNNER --output_file $OUTPUT_FILE
echo "<<<<<< Done!!! ";

