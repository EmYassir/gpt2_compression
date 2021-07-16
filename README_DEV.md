# GPT-2 Experiments*

This folder contains the original code used to train Distil* as well as examples showcasing how to use DistilGPT2. It is mainly taken and adapted from the [HuggingFace library](https://github.com/huggingface/transformers/tree/master/examples).

## Setup

This part of the library has only be tested with Python3.8+. There are few specific dependencies to install before launching a distillation, you can install them with the command `pip install -r requirements.txt`.

**Important note:** In addition to the packages present in `requirements.txt`, other additional packages need to be installed manually:
- Apex (nvidia): download [here](https://github.com/NVIDIA/apex).
- Deepspeed (Microsoft) v0.3.14: download [here](https://github.com/microsoft/DeepSpeed/releases/tag/v0.3.14).


## Available scripts

In the following, we explain how to finetune/pre-train/distil GPT2 architectures.

### A. finetune/pre-train
In order to finetune/pre-train a model, two scripts are available: `run_clm.py` and `run_clm_no_trainer.py`. In addition, a modified version of `run_clm.py`,  `custom_run_clm.py` allows for loading tokenized data (useful for large datasets). A basic example of usage would look like the following:
```bash
export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=<AN_OPEN_PORT>
export MASTER_ADDR=<I.P.>

python run_clm.py \
    --fp16 \
    --tokenizer_name gpt2 \
    --config_name config.json \
    --model_name_or_path gpt2  \
    --cache_dir <CACHE_DIRECTORY> \
    --train_file train.txt \
    --validation_file validation.txt \
    --output_dir <OUTPUT_DIRECTORY>  \
    --logging_dir <LOG_DIRECTORY> \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --do_train \
    --num_train_epochs 6 \
    --block_size 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --metric_for_best_model eval_loss \
    --evaluation_strategy "epoch"\
    --overwrite_output_dir;

```
 Some of our scripts make use of deepseed (as indicated in scripts' names) to fasten training. User can refer to the following examples:
- train.sh
- train_deepspeed.sh
- pretrain.sh
- pretrain_deepspeed.sh
- train_no_trainer.sh

### B. Tokenization before pre-training
Some datasets might be too large to be processed on the fly, so we created a script to help tokenizing the datasets and save them before training. The script `tools/tokenize_datasets.py` serves that purpose. Also, we have added options (`--tokenized_datasets` and `--grouped_tokenized_datasets`) to the script `custom_run_clm.py` (modified version of `run_clm.py`) to directly load the tokenized datasets in case they are available. An example of usage would be:

```bash
python custom_run_clm.py  \
    --tokenizer_name gpt2 \
    --model_name_or_path gpt2 \
    --train_file data/train.txt \
    --validation_file data/valid.txt \
    --tokenized_datasets <PATH_TO_TOKENIZED_DATASETS> \
    --grouped_tokenized_datasets <PATH_TO_GROUPED_DATASETS> \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8 \
    --output_dir <OUTPUT_DIR>  \
    --fp16 \ # Mixed precision training
    --do_train \
    --num_train_epochs 6 \
    --block_size 1024 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --metric_for_best_model eval_loss \
    --evaluation_strategy "epoch" \
    --overwrite_output_dir;
```


### C. Evaluation
The previous scripts can also perform evaluation. Evaluation scripts below provide examples to be inspired of:
- evaluate.sh
- evaluate_deepspeed.sh

### D. Distillation

#### i. Preparation of the data
The weights of GPT2 are trained using a concatenation of Toronto Book Corpus and English Wikipedia (same training data as the English version of BERT).

To avoid processing the data several time, we do it once and for all before the training. From now on, will suppose that you have a text file `dump.txt` which contains one sequence per line (a sequence being composed of one of several coherent sentences).

First, we will binarize the data, i.e. tokenize the data and convert each token in an index in our model's vocabulary.

```bash
python tools/binarized_data.py \
    --file_path data/dump.txt \
    --tokenizer_type gpt2 \
    --tokenizer_name gpt2 \
    --dump_file data/binarized_text
```

If need be, we can put more emphasis on rare words and count the occurrences of each tokens in the data:

```bash
python scripts/token_counts.py \
    --data_file data/binarized_text.pickle \
    --token_counts_dump data/token_counts.binarized_text.pickle \
    --vocab_size 50257
```

#### ii. Training

Training with distillation is really simple once you have pre-processed the data:

```bash
python train.py \
    --student_type gpt2 \
    --student_config training_configs/distilgpt2.json \
    --teacher_type gpt2 \
    --teacher_name gpt2 \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 \
    --freeze_pos_embs \
    --dump_path serialization_dir/my_first_training \
    --data_file data/binarized_text.pickle  \
    --token_counts data/token_counts.binarized_text.pickle \
    --force # overwrites the `dump_path` if it already exists.
```

By default, this will launch a training on a single GPU (even if more are available on the cluster). Other parameters are available in the command line, please look in `train.py` or run `python train.py --help` to list them. Here's an example that runs a distributed training on a single node having 4 GPUs:

```bash
export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=4
export WORLD_SIZE=4
export MASTER_PORT=<AN_OPEN_PORT>
export MASTER_ADDR=<I.P.>

pkill -f 'python -u train.py'

python -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
        --force \
        --gpus $WORLD_SIZE \
        --student_type gpt2 \
        --student_config training_configs/distilgpt2.json \
        --teacher_type gpt2 \
        --teacher_name gpt2 \
        --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 \
        --freeze_pos_embs \
        --dump_path serialization_dir/my_first_training \
        --data_file data/binarized_text.pickle  \
        --token_counts data/token_counts.binarized_text.pickle \
        --force
```

**Tips:** Starting distilled training with good initialization of the model weights is crucial to reach decent performance. In our experiments, we initialized our model from a few layers of the teacher (gpt2) itself! Please refer to `tools/extract.py` and `tools/extract_gpt2_layers.py` to create a valid initialization checkpoint and use `--student_pretrained_weights` argument to use this initialization for the distilled training! Below is an example on how to use `tools/extract_gpt2_layers.py`:

```bash
python extract_gpt2_layers.py  \
        --ref_model_type gpt2 \
        --config_path config.json \
        --dump_checkpoint ./model.bin \
        --layers 0 2 4 7 9 10  # Layers to select from the teacher
```

**Important note:** We have created modified versions of `distiller.py` and `train.py`, and which are `new_distiller.py` and `train_distiller_new_preprocessing.py`. The main difference is in the data pre-processing.  In the original scripts, text data is processed line-by-line, which means that some sequences might be truncated while others padded to meet certain size. In the modified versions, we concatenate the sentences into a sole sample and then split it into equal parts (size == `block_size`). Anything left at the end which does not meet the size requirement is simply dropped. Readers should note that unlike the original scripts, these don't need data the pre-processing step: it is done on the fly in the code and users only need to pass the path of the data file. An example of usage is available, please refer to `distil_custom.sh`. 



## Citation

```
@inproceedings{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  booktitle={NeurIPS EMC^2 Workshop},
  year={2019}
}
```
