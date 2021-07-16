#!/bin/sh
echo "############## Starting deepspeed distillation script... ";

#### General
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=16
export CUDA_VISIBLE_DEVICES=6,7

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TRAIN_FILE=/media/data/yassir/datasets/openwebtext/train.txt
export VALIDATION_FILE=/media/data/yassir/datasets/openwebtext/valid.txt
export TOKENIZED_DATASETS=/media/data/yassir/datasets/openwebtext/tokenized
export GROUPED_DATASETS=/media/data/yassir/datasets/openwebtext/grouped

## Model
export MODEL_PATH=/media/data/yassir/truncated_models/gpt2-alt
export MODEL_TYPE=gpt2


## Training hyper-parameters
export EPOCHS=3
export BLOCK_SIZE=1024
export BATCH_SIZE=16
export EVAL_BATCH_SIZE=8
export GRAD_ACC=4

## Training settings
export RUNNER=/home/yassir/gpt2-kd/runners/custom_run_clm.py 
export OUTPUT_DIR=/media/data/yassir/output/example

## Training
python $RUNNER \
    --tokenizer_name $MODEL_TYPE \
    --config_name $MODEL_PATH/config.json \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin  \
    --cache_dir $HF_HOME \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --tokenized_datasets $TOKENIZED_DATASETS \
    --grouped_tokenized_datasets $GROUPED_DATASETS \
    --preprocessing_num_workers $NUM_PROC \
    --dataloader_num_workers $NUM_PROC \
    --output_dir $OUTPUT_DIR  \
    --fp16 \
    --do_train \
    --num_train_epochs $EPOCHS \
    --block_size $BLOCK_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --metric_for_best_model eval_loss \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir;

echo "Done!!! ";
