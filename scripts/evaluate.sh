#!/bin/sh
echo "############## Starting evaluation script... ";
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TEST_DATA_FILE=/media/data/yassir/datasets/wikitext103/wiki.test.raw.txt


## Model
export MODEL_PATH=/media/data/yassir/output/label_smoothing/truncated_models/gpt2_0.0
export CONFIG_PATH=$MODEL_PATH/config.json
export MODEL_TYPE=gpt2

## Evaluation settings
export EVAL_BATCH_SIZE=8
export RUNNER=/home/yassir/gpt2-kd/runners/run_clm.py
export OUTPUT_DIR=$MODEL_PATH/wikitext103

## Start
python  $RUNNER  \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin \
    --config_name $CONFIG_PATH  \
    --tokenizer_name $MODEL_TYPE \
    --cache_dir $HF_HOME \
    --validation_file $TEST_DATA_FILE \
    --output_dir $OUTPUT_DIR \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --overwrite_output_dir;

echo "******************* DONE !!!";
