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
export MASTER_PORT=$((((RANDOM<<15)|RANDOM)%63001+4000))
export OMP_NUM_THREADS=2

### Models
export MODEL="gpt2"
export TEACHER_WEIGHTS=/media/data/yassir/original_models/$MODEL/
#export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/$MODEL/pytorch_model.bin
export STUDENT_WEIGHTS=/media/data/yassir/original_models/distilgpt2/pytorch_model.bin
#export STUDENT_WEIGHTS=/media/data/yassir/truncated_models/gpt2-top-emb/pytorch_model.bin
#export STUDENT_CONFIG=/media/data/yassir/truncated_models/$MODEL/config.json
export STUDENT_CONFIG=/media/data/yassir/original_models/distilgpt2/config.json
#export STUDENT_CONFIG=/media/data/yassir/truncated_models/gpt2-top-emb/config.json

### Evaluation
export SEED=42
export BATCH_SIZE=12
export CKPT_INTERVAL=1000000
export RUNNER=/home/yassir/gpt2-kd/runners/run_evaluate_samples.py
export TEMPERATURE=1.0
export ALPHA_CE_STUDENT=1.0
export ALPHA_CE_TEACHER=1.0
export ALPHA_DIV=1.0
export BLOCK_SIZE=1024


## TRAINING DATA
export DATA_FILE=/media/data/yassir/datasets/openwebtext/owt_samples_data/cleaned_plain_text_final_pass_1.gpt2.pkl
export OUTPUT_DIR=/media/data/yassir/output/samples_evaluation/dataframes
#export DATA_DIR=/home/yassir/datasets/openwebtext/owt_samples_data/
#export OUTPUT_DIR=/home/yassir/output/samples_evaluation


## START
echo ">>>>>> Starting script";
python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $RUNNER \
        --seed $SEED \
        --data_file $DATA_FILE \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --n_gpu $N_GPU_NODE \
        --student_type $MODEL \
        --teacher_type $MODEL \
        --teacher_name_or_path $TEACHER_WEIGHTS \
        --student_config $STUDENT_CONFIG \
        --student_pretrained_weights $STUDENT_WEIGHTS \
        --temperature $TEMPERATURE \
        --alpha_ce_student $ALPHA_CE_STUDENT \
        --alpha_ce_teacher $ALPHA_CE_TEACHER \
        --alpha_div $ALPHA_DIV \
        --checkpoint_interval $CKPT_INTERVAL \
        --block_size $BLOCK_SIZE \
        --force # overwrites the `dump_path` if it already exists


echo "<<<<<< Done!!! ";

