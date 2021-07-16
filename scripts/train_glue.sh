#!/bin/sh

## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"

## General
export TOKENIZERS_PARALLELISM=false

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export DATASET=super_glue
#export TASK_NAME=CoLA
export TASK_NAME=wsc

## Distributed training
export CUDA_VISIBLE_DEVICES=0

## Model
export MODEL_TYPE=gpt2
export MODEL_PATH=/media/data/yassir/truncated_models/$MODEL_TYPE


## Training hyper-parameters
export EPOCHS=3
export MAX_SEQ_LEN=128
export BATCH_SIZE=8
export EVAL_BATCH_SIZE=8
export GRAD_ACC=1
export SEED=42

## Training settings
export RUNNER=/home/yassir/gpt2-kd/runners/run_glue.py
export OUTPUT_DIR=/media/data/yassir/output/$DATASET/$TASK_NAME/original_models/$MODEL_TYPE

## Training
python $RUNNER \
    --seed $SEED \
    --tokenizer_name $MODEL_TYPE \
    --model_name_or_path $MODEL_TYPE \
    --cache_dir $HF_HOME \
    --dataset_name $DATASET \
    --dataset_config_name $TASK_NAME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --logging_dir $OUTPUT_DIR/logs \
    --do_train \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_SEQ_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --metric_for_best_model eval_loss \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir;
'
python $RUNNER \
    --fp16 \
    --seed $SEED \
    --tokenizer_name $MODEL_TYPE \
    --config_name $MODEL_PATH/config.json \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin  \
    --cache_dir $HF_HOME \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR  \
    --logging_dir $OUTPUT_DIR/logs \
    --do_train \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_SEQ_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --metric_for_best_model eval_loss \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir;
'
echo "<<<<<< Done!!! ";
