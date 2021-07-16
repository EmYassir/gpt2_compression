# GPT-2 Experiments*
This folder contains the original code used to train Distil* as well as examples showcasing how to use DistilGPT2. It is mainly taken and adapted from the [HuggingFace library](https://github.com/huggingface/transformers/tree/master/examples).

## Setup
This part of the library has only be tested with `Python3.8+`, `Pytorch 1.7.1` and `CUDA 10.1`. `Pytorch 1.7.1` can be installed using the following command:
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

There are few specific dependencies to install before launching a distillation, you can install them with the command `pip install -r requirements.txt`.

**Important note:** In addition to the packages present in `requirements.txt`, other additional packages need to be installed manually, namely:
- Apex (nvidia): download [here](https://github.com/NVIDIA/apex).


## Datasets
We use two datasets for these experiments:
1 -  Wikitext103: `.txt` files available on Lyra() at `/media/data/yassir/datasets/wikitext103/`.
2 -  Openwebtext: full raw and tokenized files (`all-merged.[txt,pkl]`) available on Lyra() at `/media/data/openwebtext/`.

For pretraining experiments, we use a slightly modified version of openwebext: We have split the original dataset into training (`train.txt`) and validation (`valid.txt`) datasets. They can be retrieved on Lyra(10.218.4.31) at `/media/data/yassir/datasets/openwebtext/`. However, in order to save time, pre-tokenized/grouped versions of this dataset exist already in Lyra(10.218.4.31) at `/media/data/yassir/datasets/openwebtext/tokenized` and `/media/data/yassir/datasets/openwebtext/grouped`.

## Models
GPT2 Models can be downloaded on the fly during training. However, they are also pre-downloaded and saved in Lyra(10.218.4.31) at `/media/data/yassir/original_models/`.

## Available scripts
In the following, we explain how to finetune/pretrain/distil GPT2 architectures. Connecting to the huggingface repo is necessary while downloading the tokenizer, but this is not possible with due to the company's proxy. Users can follow these steps to deactivate the SSL verification from the environment side:
1- If you are using a `conda` environment, go to `~/.conda/envs/<NAMEOFYOURENVIRONEMNT>/lib/python3.8/site-packages/requests/sessions.py`.
2- Search for `self.verify = True`.
3- Change it to `self.verify = False`.

The next paragraphs describe how to run the scripts.

### A. finetune
In order to finetune a model on wikitext103 (or any other dataset), two scripts are available:
- train.sh
- train_deepspeed.sh

Of course, users have to adapt the variables/paths in the scripts. The difference between the two scripts is that the second one uses deepspeed to accelerate training.

### B. Pretrain
In order to finetune a model on openwebtext (or any other dataset), two scripts are available:
- pretrain.sh
- pretrain_deepspeed.sh

Users should adapt the variables/paths in the scripts. The difference between the two scripts is that the second one uses deepspeed to accelerate training. Users can either use the already tokenized/grouped datasets at `/media/data/yassir/datasets/openwebtext/tokenized` and `/media/data/yassir/datasets/openwebtext/grouped` (available on Lyra(10.218.4.31)) or re-tokenize/re-group the dataset of their choice using the script `tokenize_datasets.py`.

**Note**: in order to reproduce our best model, please use the intial weights at `/media/data/yassir/truncated_models/gpt2-alt/`.


### C. Distillation
In order to run distillation, users should make sure that student pre-trained weights, binarized openwebtext dataset and teacher model are available. They can be retrieved at these locations:
1- Student pre-trained weights: available on Lyra(10.218.4.31) at `/media/data/yassir/truncated_models/gpt2`.
2- Teacher weights: Can be downloaded on-the-fly or found on Lyra(10.218.4.31) at `/media/data/yassir/original_models/gpt2`.
3- Binarized dataset: available on Lyra(10.218.4.31) at `/media/data/openwebtext/all-merged.pkl`.

The script to run distillation is the following:
- distil.sh

Users should however adapt the variables/paths in the scripts. The other hyper-parameters are the same ones used by HuggingFace to distil GPT2. Users are encouraged not to change them if their goal is to reproduce HuggingFace's model. 


### D. Evaluation
Evaluation scripts below provide examples to be inspired of:
- evaluate.sh
- evaluate_deepspeed.sh

Again, paths/variables should be adapted according to the situation.
