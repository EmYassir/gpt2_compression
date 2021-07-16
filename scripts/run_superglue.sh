#!/bin/sh

## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"

## General
export TOKENIZERS_PARALLELISM=false

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME


## Distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



## Settings
# CB
#export TASK_NAME=cb
#export EVAL_STEP=96
#export GPU_ID=0


# BOOLQ
export TASK_NAME=boolq
export EVAL_STEP=3537
export GPU_ID=1

# COPA
#export TASK_NAME=copa
#export EVAL_STEP=150
#export GPU_ID=2

# RTE
#export TASK_NAME=rte
#export EVAL_STEP=936
#export GPU_ID=3

# WIC
#export TASK_NAME=wic
#export EVAL_STEP=2037
#export GPU_ID=4

# WSC
#export TASK_NAME=wsc
#export EVAL_STEP=210
#export GPU_ID=5


## Model
export MODEL_TYPE=gpt2

### DistilGPT2
#export MODEL_PATH=/media/data/yassir/original_models/distilgpt2
#export OUTPUT_DIR=/media/data/yassir/output/$DATASET_NAME/$TASK_NAME/original_models/pretraining/distilgpt2


### 6-Layer GPT2 trained on clean data
export MODEL_PATH=/media/data/yassir/output/truncated_models/pretraining/$MODEL_TYPE-alt
export CONFIG_PATH=/home/yassir/gpt2-kd/training_configs/distilgpt2.json
export OUTPUT_DIR=/media/data/yassir/output/$DATASET_NAME/$TASK_NAME/truncated_models/pretraining/$MODEL_TYPE-alt

### 6-Layer GPT2 trained on full data
#export MODEL_PATH=/media/data/yassir/output/truncated_models/pretraining/$MODEL_TYPE-alt_full
#export OUTPUT_DIR=/media/data/yassir/output/$DATASET_NAME/$TASK_NAME/truncated_models/pretraining/$MODEL_TYPE-alt_full


## Training hyper-parameters
export MAX_SEQ_LEN=128
export BATCH_SIZE=8
export GRAD_ACC=1
export SEED=42

export DATASET_NAME=super_glue
export EPOCHS=30
export DATASET_DIR=/media/data/yassir/datasets/$DATASET_NAME/$TASK_NAME

## Training settings
export RUNNER=/home/yassir/gpt2-kd/runners/run_superglue.py



python $RUNNER \
    --use_gpuid $GPU_ID \
    --fp16 \
    --seed $SEED \
    --tokenizer_name $MODEL_TYPE \
    --model_type $MODEL_TYPE \
    --config_name $CONFIG_PATH \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin  \
    --cache_dir $HF_HOME \
    --data_dir $DATASET_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR  \
    --do_train \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_SEQ_LEN \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --logging_steps $EVAL_STEP \
    --log_evaluate_during_training \
    --eval_and_save_steps $EVAL_STEP \
    --save_only_best \
    --overwrite_output_dir;


'
python $RUNNER \
    --seed $SEED \
    --tokenizer_name $MODEL_TYPE \
    --config_name $MODEL_PATH/config.json \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin  \
    --cache_dir $HF_HOME \
    --data_dir $DATASET_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR  \
    --do_train \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_SEQ_LEN \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --logging_steps $EVAL_STEP \
    --log_evaluate_during_training \
    --eval_and_save_steps $EVAL_STEP \
    --overwrite_output_dir;
'
# --overwrite_cache 

echo "<<<<<< Done!!! ";
