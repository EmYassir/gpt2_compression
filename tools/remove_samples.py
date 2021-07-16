# coding=utf-8
"""
Generating samples from a given dataset
"""
import argparse
import numpy as np

def remove_bad_indexes(input_file, indexes_file, output_file = './output-file.txt'):

    print(f'-- Opening the indexes file {indexes_file} ...')
    with np.load('foo.npz') as data:
        indexes = set(data[-1])
    
    print(f'-- Opening the file {input_file} ...')
    with open(input_file, "r") as fr:
        print('-- Processing the samples ...')
        lines = (line.rstrip() for line in fr)
        print(f'-- Found {len(lines)} lines')
    
    print('-- Removing bad lines ...')
    new_lines = []
    for i, line in enumerate(lines):
        if i in indexes:
            continue
        new_lines.append(line)

    print('-- Writing out new file ...')
    with open(output_file, "w") as fw:
        for line in new_lines:
            fw.write(line+'\n')
    print(f'-- Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removal of specific samples from a given dataset."
    )
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--indexes_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    remove_bad_indexes(args.data_file, args.indexes_file, args.output_file)
