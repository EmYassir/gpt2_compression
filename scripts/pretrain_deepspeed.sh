#!/bin/sh
## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"

#### Environment
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=8

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Deepspeed
export INCLUDE="localhost:0,1,2,3,4,5,6,7"
export DS_HOSTFILE=/home/yassir/gpt2-kd/deepspeed_cfg/hostfile
export DS_CONFIG=/home/yassir/gpt2-kd/deepspeed_cfg/ds_config_2.json
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))

## Datasets
export TRAIN_FILE=/media/data/yassir/datasets/openwebtext/cleaned_plain_text_final_pass_1.txt
export TOKENIZED_DATASETS=/media/data/yassir/datasets/openwebtext/tokenized_full
export GROUPED_DATASETS=/media/data/yassir/datasets/openwebtext/grouped_full

## Model
export MODEL_TYPE=gpt2
export MODEL_PATH=/media/data/yassir/truncated_models/$MODEL_TYPE-alt


## Training settings
export LOG_INTERVAL=10000
export RUNNER=/home/yassir/gpt2-kd/runners/custom_run_clm.py
export OUTPUT_DIR=/media/data/yassir/output/pretraining/$MODEL_TYPE-alt_full

## Training hyper-parameters
export EPOCHS=4
export BLOCK_SIZE=1024
export BATCH_SIZE=16
export EVAL_BATCH_SIZE=8
export GRAD_ACC=4

echo "############## Starting deepspeed distillation script... ";
deepspeed  --hostfile $DS_HOSTFILE --include=$INCLUDE --master_port=$MASTER_PORT $RUNNER  \
    --deepspeed  $DS_CONFIG \
    --tokenizer_name $MODEL_TYPE \
    --config_name $MODEL_PATH/config.json \
    --model_name_or_path $MODEL_PATH  \
    --train_file $TRAIN_FILE \
    --cache_dir $HF_HOME \
    --tokenized_datasets $TOKENIZED_DATASETS \
    --grouped_tokenized_datasets $GROUPED_DATASETS \
    --preprocessing_num_workers $NUM_PROC \
    --dataloader_num_workers $NUM_PROC \
    --output_dir $OUTPUT_DIR  \
    --fp16 \
    --do_train \
    --num_train_epochs $EPOCHS \
    --block_size $BLOCK_SIZE \
    --logging_steps $LOG_INTERVAL \
    --save_steps $LOG_INTERVAL \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --overwrite_output_dir;

echo "Done!!! ";
