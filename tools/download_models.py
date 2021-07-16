# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before training the distilled model.
Specific to RoBERTa -> DistilRoBERTa and GPT2 -> DistilGPT2.
"""
import argparse
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download of HuggingFace's model's weights"
    )
    parser.add_argument("--model_name", default="gpt2", choices=["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--dump_checkpoint", default="./", type=str)
    args = parser.parse_args()
    print(f'Downloading {args.model_name} model...')
    #model = GPT2LMHeadModel.from_pretrained(args.model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    print(f'Saving {args.model_name} model to {args.dump_checkpoint} ...')
    #model.save_pretrained(args.dump_checkpoint)
    tokenizer.save_pretrained(args.dump_checkpoint)
    print('Done!')


