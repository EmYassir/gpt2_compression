#!/bin/sh
echo "############## Starting distillation script... ";

## DISTRIBUTED COMPUTING SETTINGS

### GENERAL
export NODE_RANK=0
export N_NODES=1
export MASTER_ADDR=127.0.0.1

### ENVIRONMENT
export CUDA_VISIBLE_DEVICES=6,7
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))
export OMP_NUM_THREADS=8

## CACHE
export TRANSFORMERS_CACHE=/media/data/yassir/huggingface/
export HF_HOME=/media/data/yassir/huggingface/

## MODELS
export MODEL="gpt2"
export TEACHER_WEIGHTS=/media/data/yassir/output/original_models/gpt2/training
export STUDENT_WEIGHTS=/home/yassir/gpt2-kd/pretrained_weights/student/initial/pytorch_model.bin
export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json

## TRAINING SETTINGS
export RUNNER=/home/yassir/gpt2-kd/runners/train_modified_distiller.py

## TRAINING DATA
export DATA_FILE=/media/data/yassir/datasets/wikitext103/wiki.train.raw.txt
export OUTPUT_DIR=/media/data/yassir/output/example

## KD HYPER-PARAMETERS
export EPOCHS=6
export BATCH_SIZE=4
export GRAD_ACC=32
export TEMPERATURE=2.0
export ALPHA_CE=0.0
export ALPHA_COS=1.0
export ALPHA_CLM=1.0
export ALPHA_MSE=0.0

## START
python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $RUNNER \
        --fp16 \
        --n_epoch $EPOCHS \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC \
        --n_gpu $N_GPU_NODE \
        --student_type $MODEL \
        --teacher_type $MODEL \
        --teacher_name $TEACHER_WEIGHTS \
        --student_config $STUDENT_CONFIG \
        --student_pretrained_weights $STUDENT_WEIGHTS \
        --temperature $TEMPERATURE \
        --alpha_ce $ALPHA_CE \
        --alpha_cos $ALPHA_COS \
        --alpha_clm $ALPHA_CLM \
        --alpha_mse $ALPHA_MSE \
        --dump_path $OUTPUT_DIR \
        --data_file $DATA_FILE \
        --force # overwrites the `dump_path` if it already exists

echo "####### Done!!! ";
