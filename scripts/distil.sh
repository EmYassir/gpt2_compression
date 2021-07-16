#!/bin/sh

## PYTHON PACKAGING
export PYTHONPATH="${PYTHONPATH}:/home/yassir/gpt2-kd/"

## DISTRIBUTED COMPUTING SETTINGS

### GENERAL
export NODE_RANK=0
export N_NODES=1
export MASTER_ADDR=127.0.0.1

### ENVIRONMENT
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export N_GPU_NODE=8
export WORLD_SIZE=8
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+2000))
export OMP_NUM_THREADS=8

## CACHE
export TRANSFORMERS_CACHE=/media/data/yassir/huggingface/
export HF_HOME=/media/data/yassir/huggingface/

## MODELS
export MODEL="gpt2"
export TEACHER_NAME=gpt2
export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/gpt2/pytorch_model.bin
export STUDENT_CONFIG=/home/yassir/gpt2-kd/training_configs/distilgpt2.json

## TRAINING SETTINGS
export RUNNER=/home/yassir/gpt2-kd/runners/train.py

## TRAINING DATA
export EXP=0
#export DATA_FILE=/media/data/yassir/datasets/wikitext103/gpt2.binarized.wiki.train.raw.pickle
export DATA_FILE=/media/nvme/yassir/datasets/openwebtext/cleaned_plain_text_final_pass_1.gpt2.pkl
export OUTPUT_DIR=/media/data/yassir/output/deepspeed_distillation/exp-$EXP

## KD HYPER-PARAMETERS
export EPOCHS=4
#export BATCH_SIZE=1
#export GRAD_ACC=500
export BATCH_SIZE=4
export GRAD_ACC=128
export TEMPERATURE=2.0
export ALPHA_CE=5.0
export ALPHA_COS=1.0
export ALPHA_CLM=2.0
export ALPHA_MSE=0.0
export WARMUP_PROP=0.05
export WEIGHT_DECAY=0.0
export LEARNING_RATE=0.00025
export ADAM_EPSILON=1e-6
export MAX_GRAD_NORM=5.0
export INIT_RANGE=0.02


echo "############## Starting distillation script... ";

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
        --teacher_name $TEACHER_NAME \
        --student_config $STUDENT_CONFIG \
        --student_pretrained_weights $STUDENT_WEIGHTS \
        --temperature $TEMPERATURE \
        --alpha_ce $ALPHA_CE \
        --alpha_cos $ALPHA_COS \
        --alpha_clm $ALPHA_CLM \
        --alpha_mse $ALPHA_MSE \
        --warmup_prop $WARMUP_PROP \
        --weight_decay $WEIGHT_DECAY \
        --learning_rate $LEARNING_RATE \
        --adam_epsilon $ADAM_EPSILON \
        --max_grad_norm $MAX_GRAD_NORM \
        --initializer_range $INIT_RANGE \
        --freeze_pos_embs \
        --dump_path $OUTPUT_DIR \
        --data_file $DATA_FILE \
        --force # overwrites the `dump_path` if it already exists

echo "<<<<<< Done!!! ";
