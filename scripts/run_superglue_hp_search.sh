#!/bin/sh

## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"

## General
export TOKENIZERS_PARALLELISM=false

## CACHE
#export HF_HOME=/media/data/yassir/huggingface/
export HF_HOME=/home/yassir/.cache/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME


## Distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

## Disable Tune callbacks that cause errors
export TUNE_DISABLE_AUTO_CALLBACK_LOGGERS=1




## Training hyper-parameters
export MAX_SEQ_LEN=128
export BATCH_SIZE=8
export GRAD_ACC=1
export SEED=42
export TEMPERATURE=1.
export ALPHA_CE=0.2
export ALPHA_ATT=0.2
export ALPHA_VAL=0.2
export ALPHA_PKD=0.2
export BETA_PKD=0.2
export PKD_OUTPUT=0


## Settings
export TASK_NAME=cb
export DATASET_NAME=super_glue
export EPOCHS=30
export EVAL_STEP=96
export GPU_ID=-1
#export DATASET_DIR=/media/data/yassir/datasets/$DATASET_NAME/$TASK_NAME
export DATASET_DIR=/home/yassir/datasets/$DATASET_NAME/$TASK_NAME


## Model
export MODEL_TYPE=gpt2
#export TEACHER_WEIGHTS=/media/data/yassir/output/$DATASET_NAME/$TASK_NAME/original_models/$MODEL_TYPE/checkpoint-best
#export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/$MODEL_TYPE/pytorch_model.bin
#export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json
## Adapt paths for maple
export TEACHER_WEIGHTS=/home/yassir/models/$DATASET_NAME/$TASK_NAME/original_models/$MODEL_TYPE/checkpoint-best
export STUDENT_WEIGHTS=/home/yassir/models/student/pytorch_model.bin
export STUDENT_CONFIG=/home/yassir/models/student/config.json

## Training settings
export OPTION1=minilm
export OPTION2=pkd
export RUNNER=/home/yassir/gpt2-kd/runners/run_hp_tune_superglue.py
#export OUTPUT_DIR=/media/nvme/yassir/output/hp_search/$DATASET_NAME/$TASK_NAME/truncated_models/$MODEL_TYPE
export OUTPUT_DIR=/home/yassir/output/hp_search/$DATASET_NAME/$TASK_NAME/truncated_models/$MODEL_TYPE

python $RUNNER \
    --use_gpuid $GPU_ID \
    --seed $SEED \
    --data_dir $DATASET_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --config_name $STUDENT_CONFIG \
    --tokenizer_name $MODEL_TYPE \
    --model_type $MODEL_TYPE \
    --model_name_or_path $STUDENT_WEIGHTS \
    --teacher_type $MODEL_TYPE \
    --teacher_name_or_path $TEACHER_WEIGHTS \
    --temperature $TEMPERATURE \
    --alpha_ce $ALPHA_CE \
    --alpha_att $ALPHA_ATT \
    --alpha_val $ALPHA_VAL \
    --$OPTION1 \
    --$OPTION2 \
    --alpha_pkd $ALPHA_PKD \
    --beta_pkd $BETA_PKD \
    --pkd_output $PKD_OUTPUT \
    --cache_dir $HF_HOME \
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
    --overwrite_output_dir \
    --teacher_layers 0 2 4 7 9 11

# --overwrite_cache 

echo "<<<<<< Done!!! ";
