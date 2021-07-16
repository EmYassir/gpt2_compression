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
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+3000))

## CACHE
export TRANSFORMERS_CACHE=/media/data/yassir/huggingface/
export HF_HOME=/media/data/yassir/huggingface/

## MODELS
export MODEL="gpt2"
export TEACHER_WEIGHTS=/media/data/yassir/output/original_models/$MODEL/training
export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/$MODEL/pytorch_model.bin
export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json

## TRAINING SETTINGS
export EXP=1
export OPTION1=minilm
export OPTION2=pkd
export RUNNER=/home/yassir/gpt2-kd/runners/train_pkd_distiller.py
#export OUTPUT_DIR=/media/data/yassir/output/pkd_distillation/$MODEL/$OPTION\_$EXP
#export OUTPUT_DIR=/media/data/yassir/output/pkd_distillation/$MODEL/$OPTION1\_$OPTION2\_$EXP
export OUTPUT_DIR=/media/data/yassir/output/pkd_distillation/test-3/

## TRAINING DATA
export DATA_DIR=/media/data/yassir/datasets/wikitext103/grouped

## KD HYPER-PARAMETERS
export EPOCHS=6
export BATCH_SIZE=8
export GRAD_ACC=4
export TEMPERATURE=1.
export ALPHA_LM=0.2
export ALPHA_ATT=0.2
export ALPHA_VAL=0.2
export ALPHA_PKD=0.2
export BETA_PKD=0.2
export PKD_OUTPUT=0


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
    --teacher_name_or_path $TEACHER_WEIGHTS \
    --student_config $STUDENT_CONFIG \
    --student_pretrained_weights $STUDENT_WEIGHTS \
    --temperature $TEMPERATURE \
    --alpha_lm $ALPHA_LM \
    --alpha_att $ALPHA_ATT \
    --alpha_val $ALPHA_VAL \
    --$OPTION1 \
    --$OPTION2 \
    --beta_pkd $BETA_PKD \
    --pkd_output $PKD_OUTPUT \
    --data_dir $DATA_DIR \
    --dump_path $OUTPUT_DIR \
    --force \
    --teacher_layers 0 2 4 7 9 11 

echo "<<<<<< Done!!! ";

