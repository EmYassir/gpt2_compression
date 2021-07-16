#!/bin/sh

## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"


## General
export TOKENIZERS_PARALLELISM=false


## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME


## settings
export DATASET_NAME=super_glue
export TASK_NAME=record
export EPOCHS=10
export EVAL_STEP=7019
export GPU_ID=-1
export DATASET_DIR=/media/data/yassir/datasets/$DATASET_NAME/$TASK_NAME


## Distributed training (or not)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


## Model
## MODELS
export MODEL_TYPE="gpt2"
#export TEACHER_WEIGHTS=/media/data/yassir/output/$DATASET_NAME/$TASK_NAME/original_models/$MODEL_TYPE/checkpoint-best
#export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/$MODEL_TYPE/pytorch_model.bin
#export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json
export TEACHER_WEIGHTS=/media/data/yassir/output/$DATASET_NAME/$TASK_NAME/original_models/$MODEL_TYPE/checkpoint-best
export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/$MODEL_TYPE/pytorch_model.bin
export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json


## Training hyper-parameters
export BATCH_SIZE=32
export MAX_SEQ_LEN=128
export GRAD_ACC=2
export SEED=42
export TEMPERATURE=1.
export ALPHA_CE=0.2
export ALPHA_ATT=0.2
export ALPHA_VAL=0.2
export ALPHA_PKD=0.2
export BETA_PKD=0.2
export PKD_OUTPUT=0


## Training settings
export EXP=8
export OPTION1=minilm
export OPTION2=pkd
export RUNNER=/home/yassir/gpt2-kd/runners/run_superglue_distillation.py
export OUTPUT_DIR=/media/data/yassir/output/distillation/$DATASET_NAME/$TASK_NAME/$EXP/truncated_models/$MODEL_TYPE


## Training
python $RUNNER \
    --use_gpuid $GPU_ID \
    --seed $SEED \
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
    --data_dir $DATASET_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_SEQ_LEN \
    --per_gpu_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --do_eval \
    --logging_steps $EVAL_STEP \
    --log_evaluate_during_training \
    --eval_and_save_steps $EVAL_STEP \
    --overwrite_output_dir \
    --teacher_layers 0 2 4 7 9 11


echo "<<<<<< Done!!! ";
