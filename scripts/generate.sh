#!/bin/sh
echo "############## Starting generation script... ";
export NODE_RANK=0
export N_NODES=1
export N_GPU_NODE=8
export WORLD_SIZE=8
export MASTER_ADDR=127.0.0.1
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=/media/data/yassir/huggingface/
export HF_HOME=/media/data/yassir/huggingface/

export N_EX=10

#### Pretrained models
export MODEL_TYPE=gpt2
export MODEL_PATH=/media/data/yassir/output/truncated_models/pretraining/$MODEL_TYPE

#### Settings
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE
export RUNNER=/home/yassir/gpt2-kd/runners/run_generation.py


echo "******************* generating sentences with "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done


echo "******************* DONE !!!";

export MODEL_TYPE=distilgpt2
export MODEL_PATH=/media/data/yassir/original_models/$MODEL_TYPE
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE

echo "******************* generating sentences with "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done
echo "******************* DONE !!!";

#### Pretrained models
export MODEL_TYPE=gpt2
export MODEL_PATH=/media/data/yassir/original_models/$MODEL_TYPE/
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE-small

echo "******************* generating sentences with pretrained "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done


export MODEL_TYPE=gpt2-xl
export MODEL_PATH=/media/data/yassir/original_models/$MODEL_TYPE/
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE

echo "******************* generating sentences with  "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done



#### Finetuned models

export MODEL_TYPE=gpt2
export MODEL_PATH=/media/data/yassir/output/pretraining/$MODEL_TYPE/finetuned-weights/
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE-finetuned

echo "******************* generating sentences with finetuned "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done

export MODEL_TYPE=distilgpt2
export MODEL_PATH=/media/data/yassir/output/original_models/$MODEL_TYPE/training/
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE-finetuned

echo "******************* generating sentences with finetuned "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done

export MODEL_TYPE=gpt2
export MODEL_PATH=/media/data/yassir/output/original_models/$MODEL_TYPE/training/
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE-small-finetuned

echo "******************* generating sentences with finetuned "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done


export CUDA_VISIBLE_DEVICES=3
export MODEL_TYPE=gpt2-xl
export MODEL_PATH=/media/data/yassir/output/original_models/$MODEL_TYPE/training/
export OUT_DIR=/media/data/yassir/output/generation/$MODEL_TYPE-finetuned

echo "******************* generating sentences with finetuned "$MODEL_TYPE" ...";
for PROMPT in "A" "The" "I" "Nowadays" "Jesus" "Today" "Usually" "I want" "I believe that" "I enjoy walking with my cute dog" "I knew this information could be life-saving, I felt like I was being a language activist"
do
    for T in 0.5 0.7 1 1.5 2 2.5
    do
        for SEQ_LENGTH in 10 20 30 40
        do
	        python  $RUNNER  --model_type gpt2 --model_name_or_path $MODEL_PATH  --temperature $T --prompt "$PROMPT" --length $SEQ_LENGTH --seed $SEQ_LENGTH --num_return_sequences $N_EX --output_dir $OUT_DIR/T$T/$SEQ_LENGTH
        done
    done
done

