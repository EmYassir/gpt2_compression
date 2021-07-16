#!/bin/sh

## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"

## General
#export NODE_RANK=0
#export N_NODES=1
#export N_GPU_NODE=2
#export WORLD_SIZE=2
#export MASTER_ADDR=127.0.0.1
#export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export DATASET=super_glue
#export TASK_NAME=CoLA
export TASK_NAME=boolq
export VALIDATION_FILE=/media/data/yassir/datasets/super_glue/$TASK_NAME/val.jsonl


## Distributed training
export CUDA_VISIBLE_DEVICES=1

## Training hyper-parameters
export MAX_SEQ_LEN=128
export EVAL_BATCH_SIZE=8
export SEED=42

## Training settings
export MODEL_TYPE=gpt2
export MODEL_PATH=/media/data/yassir/output/$DATASET/$TASK/original_models/$MODEL_TYPE
export OUTPUT_DIR=$MODEL_PATH/eval
export RUNNER=/home/yassir/gpt2-kd/runners/run_glue.py

## Training
python $RUNNER \
    --seed $SEED \
    --tokenizer_name $MODEL_TYPE \
    --config_name $MODEL_PATH/config.json \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin  \
    --cache_dir $HF_HOME \
    --task_name $TASK_NAME \
    --validation_file $VALIDATION_FILE \
    --output_dir $OUTPUT_DIR  \
    --logging_dir $OUTPUT_DIR/logs \
    --max_seq_length $MAX_SEQ_LEN \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --overwrite_output_dir;

echo "<<<<<< Done!!! ";
