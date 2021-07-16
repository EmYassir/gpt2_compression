#!/usr/bin/env bash

function runexp {
export CUDA_VISIBLE_DEVICES=${2}
export GLUE_DIR=/home/tianda/Zero_shot/data/super_glue
export TASK_NAME=${1}

gpu=${2}      # The GPU you want to use
mtype=${3}     # Model type
mname=${4}    # Model name
#alr=${4}      # Step size of gradient ascent
#amag=${5}     # Magnitude of initial (adversarial?) perturbation
#anorm=${6}    # Maximum norm of adversarial perturbation
#asteps=${7}   # Number of gradient ascent steps for the adversary
lr=${5}       # Learning rate for model parameters
bsize=${6}    # Batch size
gas=${7}     # Gradient accumulation. bsize * gas = effective batch size
seqlen=128    # Maximum sequence length
epo=${8}      # Number of training epochs (counted as parameter updates)
wr=${9}      # Learning rate warm-up steps
seed=${10}    # Seed for randomness
wd=${11}      # Weight decay

expname=baseline-${mname}-${TASK_NAME}-sl${seqlen}-lr${lr}-bs${bsize}-gas${gas}-ts${epo}-ws${wr}-wd${wd}-seed${seed}
#checkpoint_dir=/home/tianda/gradient-augmentation-kd-master/transformers-master/glue/BERT_base
checkpoint_dir=../output

python run_superglue.py \
  --model_type ${mtype} \
  --model_name_or_path gpt2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --overwrite_output_dir \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --num_train_epochs 30\
  --max_seq_length ${seqlen} \
  --per_gpu_train_batch_size ${bsize} --gradient_accumulation_steps ${gas} \
  --learning_rate ${lr} --weight_decay ${wd} \
  --output_dir ${checkpoint_dir}/${TASK_NAME}/ \
  --logging_steps 10 --eval_and_save_steps 10 
#> ../log/Fintune_${TASK_NAME}_gpt.log 2>&1 &
}


# runexp TASK_NAME  gpu   model_type   model_name   lr     bsize  grad_accu       epo     wr     seed      wd
#runexp  boolq       0     gpt2           gpt2     2e-5      8       1             10     0.1     50     1e-2
#runexp  cb          2     gpt2           gpt2     2e-5      8       1             10     0.1     50     1e-2
runexp  copa        2     gpt2           gpt2     2e-5      8       1             10     0.1     50     1e-2
#runexp  multirc     2     gpt2           gpt2     2e-5      8       1             4      0.1     50     1e-2
#runexp  record      5     gpt2           gpt2     2e-5      8       1             4      0.1     50     1e-2
#runexp  rte         6     gpt2           gpt2     2e-5      8       1             10     0.1     50     1e-2
#runexp  wic         7     gpt2           gpt2     2e-5      8       1             10     0.1     50     1e-2
#runexp  wsc         1     gpt2           gpt2     2e-5      8       1             10     0.1     50     1e-2


