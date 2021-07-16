#!/bin/bash
echo "############## Starting deepspeed evaluation script... ";
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

## Deepspeed
export EXCLUDE="localhost:0,1,2,3,4,5"
export DS_HOSTFILE=/home/yassir/gpt2-kd/deepspeed_cfg/hostfile
export DS_CONFIG=/home/yassir/gpt2-kd/deepspeed_cfg/ds_config_2.json
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TEST_DATA_FILE=/media/data/yassir/datasets/wikitext103/wiki.test.raw.txt

## Model 
export MODEL_PATH=/media/data/yassir/truncated_models/gpt2-alt
export MODEL_TYPE=gpt2

## Evaluation settings
export EVAL_BATCH_SIZE=8
export RUNNER=/home/yassir/gpt2-kd/runners/run_clm.py

## Start
deepspeed  --hostfile $DS_HOSTFILE --exclude=$EXCLUDE --master_port=$MASTER_PORT  $RUNNER \
    --deepspeed  $DS_CONFIG \
    --tokenizer_name $MODEL_TYPE \
    --config_name $MODEL_PATH/config.json \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin \
    --validation_file $TEST_DATA_FILE \
    --output_dir /media/data/yassir/output/truncated_models/$MODEL/output \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --overwrite_output_dir;

echo "******************* DONE !!!";
