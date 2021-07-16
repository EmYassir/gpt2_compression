# coding=utf-8
"""
Generating samples from a given dataset
"""
import argparse
import numpy as np

def sample_from_dataset(input_file, output_dir='./', seed=None, p=0.15, valid=0.05):
  if p > 1:
    p = (p % 100)/100.
  with open(input_file, "r") as fr:
    lines = (line.rstrip() for line in fr) 
    lines = list(line for line in lines if line) # Non-blank lines in a list
    l = len(lines)
    print('Read %d lines' %l)
    np.random.seed(seed=seed)
    #indexes = np.sort(np.random.choice(l, int(p*l), replace=False))
    indexes = np.random.choice(l, int(p*l), replace=False)
    split_index = int(len(indexes)*(1 - valid))
    index_valid = indexes[split_index:]
    index_train = indexes[0:split_index]
    print('Choosing %d lines for training and %d lines for validation ...' %(len(index_train), len(index_valid)))
    train_set = [lines[i] for i in index_train]
    valid_set = [lines[i] for i in index_valid]
  with open(output_dir + 'train.txt', "w") as fw:
      print(f"Saving train set to {output_dir + 'train.txt'}...")
      for line in train_set:
          fw.write(line+'\n')
  with open(output_dir + 'valid.txt', "w") as fw:
      print(f"Saving validation set to {output_dir + 'valid.txt'}...")
      for line in valid_set:
          fw.write(line+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sampling from a given dataset."
    )
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", default="./", type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--percent", default=0.15, type=float)
    parser.add_argument("--valid", default=0.05, type=float)
    args = parser.parse_args()
    sample_from_dataset(args.input_file, args.output_dir, seed=args.seed, p=args.percent, valid=args.valid)
