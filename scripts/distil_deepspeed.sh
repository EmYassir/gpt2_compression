#!/bin/sh

## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"

## DISTRIBUTED COMPUTING SETTINGS
## Workaround to avoid deadlock
export NCCL_LL_THRESHOLD=0

### GENERAL
export NODE_RANK=0
export N_NODES=1
export MASTER_ADDR=127.0.0.1

### ENVIRONMENT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPU_NODE=8
export WORLD_SIZE=8
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+3000))
export OMP_NUM_THREADS=8

## Deepspeed 
export INCLUDE="localhost:0,1,2,3,4,5,6,7"
export DS_HOSTFILE=/home/yassir/gpt2-kd/deepspeed_cfg/hostfile
export DS_CONFIG=/home/yassir/gpt2-kd/deepspeed_cfg/ds_config_2.json

## CACHE
export TRANSFORMERS_CACHE=/media/data/yassir/huggingface/
export HF_HOME=/media/data/yassir/huggingface/

## MODELS
export MODEL="gpt2"
export TEACHER_WEIGHTS=/media/data/yassir/original_models/$MODEL/
export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/$MODEL/pytorch_model.bin
export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json
export TOKENIZER_PATH=/media/data/yassir/tokenizer/

## TRAINING SETTINGS
export EXP=1
export RUNNER=/home/yassir/gpt2-kd/runners/train_deepspeed.py
export OUTPUT_DIR=/media/data/yassir/output/deepspeed_distillation/exp-$EXP

## TRAINING DATA
export DATA_FILE=/media/nvme/yassir/datasets/openwebtext/cleaned_plain_text_final_pass_1.gpt2.pkl

## KD HYPER-PARAMETERS
export EPOCHS=4
export BATCH_SIZE=8
export GRAD_ACC=64
export TEMPERATURE=2.
export ALPHA_CE=5.0
export ALPHA_COS=1.0
export ALPHA_CLM=2.0
export ALPHA_MSE=0.0


## START
echo ">>>>>> Starting distillation";
deepspeed  --hostfile $DS_HOSTFILE --include=$INCLUDE --master_port=$MASTER_PORT  $RUNNER \
    --deepspeed $DS_CONFIG \
    --fp16 \
    --n_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --n_gpu $N_GPU_NODE \
    --student_type $MODEL \
    --teacher_type $MODEL \
    --tokenizer_path $TOKENIZER_PATH \
    --teacher_name_or_path $TEACHER_WEIGHTS \
    --student_config $STUDENT_CONFIG \
    --student_pretrained_weights $STUDENT_WEIGHTS \
    --temperature $TEMPERATURE \
    --alpha_ce $ALPHA_CE \
    --alpha_cos $ALPHA_COS \
    --alpha_clm $ALPHA_CLM \
    --alpha_mse $ALPHA_MSE \
    --dump_path $OUTPUT_DIR \
    --data_file $DATA_FILE \
    --freeze_pos_embs \
    --force \
    --teacher_layers 0 2 4 7 9 11 

echo "<<<<<< Done!!! ";

