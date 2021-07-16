echo "############## Starting distillation script... ";

## General
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=2
export WORLD_SIZE=2
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=8

## CACHE
export HF_HOME=/media/data/yassir/huggingface/
export TRANSFORMERS_CACHE=$HF_HOME

## Datasets
export TRAIN_FILE=/media/data/yassir/datasets/wikitext103/wiki.train.raw.txt
export VALIDATION_FILE=/media/data/yassir/datasets/wikitext103/wiki.valid.raw.txt

## Distributed training
export CUDA_VISIBLE_DEVICES=6,7

## Model
export MODEL_PATH=/media/data/yassir/truncated_models/gpt2-alt
export MODEL_TYPE=gpt2

## Training settings
export RUNNER=/home/yassir/gpt2-kd/runners/run_clm_no_trainer.py

## Training parameters
export EPOCHS=6
export BLOCK_SIZE=1024
export BATCH_SIZE=16
export EVAL_BATCH_SIZE=8
export GRAD_ACC=8
export OUTPUT_DIR=/media/data/yassir/output/example

echo "==========>>>>>>>>> Training gpt2...";

python $RUNNER  \
    --tokenizer_name $MODEL_TYPE \
    --config_name $MODEL_PATH/config.json \
    --model_name_or_path $MODEL_PATH/pytorch_model.bin  \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --output_dir $OUTPUT_DIR  \
    --preprocessing_num_workers $NUM_PROC \
    --num_train_epochs $EPOCHS \
    --block_size $BLOCK_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE;

echo "<<<<<< Done!!! ";
