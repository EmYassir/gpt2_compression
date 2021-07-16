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
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))
export OMP_NUM_THREADS=2

## Deepspeed 
export INCLUDE="localhost:0,1,2,3,4,5,6,7"
export DS_HOSTFILE=/home/yassir/gpt2-kd/deepspeed_cfg/hostfile
export DS_CONFIG=/home/yassir/gpt2-kd/deepspeed_cfg/ds_config_2.json

## CACHE
export TRANSFORMERS_CACHE=/media/nvme/yassir/huggingface/
export HF_HOME=/media/nvme/yassir/huggingface/

## MODELS
export MODEL="gpt2"
export TEACHER_WEIGHTS=/media/data/yassir/original_models/$MODEL/
export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/$MODEL/pytorch_model.bin
export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json

## TRAINING SETTINGS
export EXP=8
export OPTION1=minilm
export OPTION2=pkd
export RUNNER=/home/yassir/gpt2-kd/runners/train_pkd_dist_hf_preproc.py
export OUTPUT_DIR=/media/nvme/yassir/output/pkd_distillation/pretraining/exp-$EXP

## TRAINING DATA
export DATA_FILE=/media/nvme/yassir/datasets/openwebtext/cleaned_plain_text_final_pass_1.gpt2.pkl

## KD HYPER-PARAMETERS
export EPOCHS=4
export BATCH_SIZE=8
export GRAD_ACC=64
export TEMPERATURE=1.
export ALPHA_LM=0.2
export ALPHA_ATT=0.2
export ALPHA_VAL=0.2
export ALPHA_PKD=0.2
export BETA_PKD=0.2
export PKD_OUTPUT=0

## START
echo ">>>>>> Starting distillation";
deepspeed --hostfile $DS_HOSTFILE --include=$INCLUDE --master_port=$MASTER_PORT  $RUNNER \
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
    --alpha_lm $ALPHA_LM \
    --alpha_att $ALPHA_ATT \
    --alpha_val $ALPHA_VAL \
    --$OPTION1 \
    --$OPTION2 \
    --beta_pkd $BETA_PKD \
    --pkd_output $PKD_OUTPUT \
    --data_file $DATA_FILE \
    --dump_path $OUTPUT_DIR \
    --force \
    --teacher_layers 0 2 4 7 9 11 

echo "<<<<<< Done!!! ";
