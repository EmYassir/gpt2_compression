# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
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
""" Dataset to distilled models
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from utilities.utils import logger
import pickle as pkl

class LmSeqsDataset(Dataset):
    """Custom Dataset wrapping language modeling sequences.

    Each sample will be retrieved by indexing the list of token_ids and their corresponding lengths.

    Input:
    ------
        params: `NameSpace` parameters
        data: `List[np.array[int]]
    """
    def __init__(self, params, max_model_input_size, special_tok_ids, **kwargs):
        self.params = params
        self.max_model_input_size = max_model_input_size
        self.special_tok_ids = special_tok_ids
        if len(kwargs) == 1:
            data = kwargs['data']
            """
            self.token_ids = np.array(data, dtype=np.uint16)
            self.lengths = np.array([len(t) for t in data], dtype=np.uint16)
            self.indexes = np.arange(len(self.token_ids), dtype=np.uint32)
            """
            self.token_ids = np.array(data, dtype=object)
            self.lengths = np.array([len(t) for t in data])
            self.indexes = np.arange(len(self.token_ids))
            self.check()
            self.remove_long_sequences()
            self.index_sentences()
            self.remove_empty_sequences()
            self.index_sentences()
            self.remove_unknown_sequences()
            self.index_sentences()
            self.check()
        else:
            protocol, data_dir = kwargs['protocol'], kwargs['data_dir']
            self.load_data(as_format = protocol, data_dir = data_dir)
            self.check()
        logger.info(f"Final data length == {len(self.token_ids)}")

    def __getitem__(self, index):
        return (self.indexes[index], self.token_ids[index], self.lengths[index])

    def __len__(self):
        return len(self.lengths)

    def check(self):
        """
        Some sanity checks
        """
        assert len(self.indexes) == len(self.token_ids)
        assert len(self.token_ids) == len(self.lengths)
        assert all(self.lengths[i] == len(self.token_ids[i]) for i in range(len(self.lengths)))
    
    def index_sentences(self):
        #self.indexes = np.arange(len(self.token_ids), dtype=np.uint32)
        self.indexes = np.arange(len(self.token_ids))

    def remove_long_sequences(self):
        """
        Sequences that are too long are split by chunk of max_model_input_size.
        """
        max_len = self.max_model_input_size
        indices = self.lengths > max_len
        logger.info(f"Splitting {sum(indices)} too long sequences.")

        def divide_chunks(l, n):
            return [l[i : i + n] for i in range(0, len(l), n)]

        new_tok_ids = []
        new_lengths = []
        cls_id, sep_id = self.special_tok_ids["bos_token"], self.special_tok_ids["eos_token"]

        for seq_, len_ in zip(self.token_ids, self.lengths):
            assert (seq_[0] == cls_id) and (seq_[-1] == sep_id), seq_
            if len_ <= max_len:
                new_tok_ids.append(seq_)
                new_lengths.append(len_)
            else:
                sub_seqs = []
                for sub_s in divide_chunks(seq_, max_len - 2):
                    if sub_s[0] != cls_id:
                        sub_s = np.insert(sub_s, 0, cls_id)
                    if sub_s[-1] != sep_id:
                        sub_s = np.insert(sub_s, len(sub_s), sep_id)
                    assert len(sub_s) <= max_len
                    assert (sub_s[0] == cls_id) and (sub_s[-1] == sep_id), sub_s
                    sub_seqs.append(sub_s)

                new_tok_ids.extend(sub_seqs)
                new_lengths.extend([len(l) for l in sub_seqs])
        """
        self.token_ids = np.array(new_tok_ids, dtype=np.uint16)
        self.lengths = np.array(new_lengths, dtype=np.uint16)
        """
        self.token_ids = np.array(new_tok_ids)
        self.lengths = np.array(new_lengths)
        logger.info(f"Got {len(new_tok_ids)} sequences...")

    def truncate_long_sequences(self):
        """
        Sequences that are too long are truncated not to exceed max_model_input_size in length. 
        The remaining chunks are simply dropped.
        """
        max_len = self.max_model_input_size
        indices = self.lengths > max_len
        logger.info(f"Truncating {sum(indices)} too long sequences.")

        def truncate_chunk(l, n):
            return l[:n]

        new_tok_ids = []
        new_lengths = []
        cls_id, sep_id = self.special_tok_ids["bos_token"], self.special_tok_ids["eos_token"]
        for i, (seq_, len_) in enumerate(zip(self.token_ids, self.lengths)):
            assert (seq_[0] == cls_id) and (seq_[-1] == sep_id), seq_
            if len_ <= max_len:
                new_tok_ids.append(seq_)
                new_lengths.append(len_)
            else:
                sub_s = truncate_chunk(seq_, max_len)
                sub_s[-1] = sep_id
                assert len(sub_s) <= max_len
                assert (sub_s[0] == cls_id) and (sub_s[-1] == sep_id), sub_s
                new_tok_ids.append(sub_s)
                new_lengths.append(len(sub_s))
        """
        self.token_ids = np.array(new_tok_ids, dtype=np.uint16)
        self.lengths = np.array(new_lengths, dtype=np.uint16)
        """
        self.token_ids = np.array(new_tok_ids)
        self.lengths = np.array(new_lengths)

    def remove_empty_sequences(self):
        """
        Too short sequences are simply removed. This could be tuned.
        """
        init_size = len(self)
        indices = self.lengths > 11
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} too short (<=11 tokens) sequences.")
        logger.info(f"kept {new_size} sequences...")


    def remove_unknown_sequences(self):
        """
        Remove sequences with a (too) high level of unknown tokens.
        """
        if "unk_token" not in self.special_tok_ids:
            return
        else:
            unk_token_id = self.special_tok_ids["unk_token"]
        init_size = len(self)
        #unk_occs = np.array([np.count_nonzero(a == unk_token_id) for a in self.token_ids], dtype=np.uint16)
        unk_occs = np.array([np.count_nonzero(a == unk_token_id) for a in self.token_ids])
        indices = (unk_occs / self.lengths) < 0.5
        self.token_ids = self.token_ids[indices]
        self.lengths = self.lengths[indices]
        new_size = len(self)
        logger.info(f"Remove {init_size - new_size} sequences with a high level of unknown tokens (50%).")


    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """
        indexes = [t[0] for t in batch]
        token_ids = [t[1] for t in batch]
        lengths = [t[2] for t in batch]
        assert len(token_ids) == len(lengths)

        # Max for paddings
        max_seq_len_ = max(lengths)

        # Pad token ids
        pad_idx = self.special_tok_ids["unk_token"]
        tk_ = [list(t.astype(int)) + [pad_idx] * (max_seq_len_ - len(t)) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        ind_t = torch.tensor(indexes)
        tk_t = torch.tensor(tk_)  # (bs, max_seq_len_)
        lg_t = torch.tensor(lengths)  # (bs)
        return ind_t, tk_t, lg_t

    
    def get_data(self, as_tensor = False):
        if as_tensor == True:
            return torch.tensor(self.indexes), torch.tensor(self.token_ids), torch.tensor(self.lengths)
        else:
            return self.indexes, self.token_ids, self.lengths
    
    def save_data(self, as_format = 'h5', dump_dir = './'):
        if as_format == 'np':
            filepath = os.path.join(dump_dir, "arrays.npz")
            np.savez(filepath, self.indexes, self.token_ids, self.lengths)
        elif as_format == 'ts':
            cfg = {'indexes.bin': self.indexes, 'tokens.bin': self.token_ids, 'lengths.bin': self.lengths}
            for filename, arr in cfg.items():
                torch.save(torch.tensor(arr.astype(int)), os.path.join(dump_dir, filename))
        elif as_format == 'pkl':
            filepath = os.path.join(dump_dir, "arrays.pkl")
            with open(filepath, "wb") as fp:
                pkl.dump((self.indexes, self.token_ids, self.lengths), fp)
        else:
            filepath = os.path.join(dump_dir, "arrays.h5")
            hf = h5py.File(filepath, 'w')
            hf.create_dataset('indexes', data=self.indexes)
            hf.create_dataset('lengths', data=self.lengths)
            ## Flatten the array
            logger.info('Flattening...')
            flat_token_ids = []
            flat_token_ids.extend([row for row in self.token_ids])
            hf.create_dataset('token_ids', data=flat_token_ids)
            hf.close()
            
        print('Done!!!')

    def load_data(self, as_format = 'h5', data_dir = './'):
        print(f'Loading data with protocol {as_format} ...')
        if as_format == 'np':
            filepath = os.path.join(data_dir, "arrays.npz")
            self.indexes, self.token_ids, self.lengths = np.load(filepath)
        elif as_format == 'ts':
            cfg = {'indexes.bin': self.indexes, 'tokens.bin': self.token_ids, 'lengths.bin': self.lengths}
            for filename, arr in cfg.items():
                torch.save(torch.tensor(arr.astype(int)), os.path.join(data_dir, filename))
        elif as_format == 'pkl':
            filepath = os.path.join(data_dir, "arrays.pkl")
            with open(filepath, "rb") as fp:
                (self.indexes, self.token_ids, self.lengths) = pkl.load(fp)
            self.indexes = np.array(self.indexes)
            self.token_ids = np.array(self.token_ids, dtype=object)
            self.lengths = np.array(self.lengths)
        else:
            filepath = os.path.join(data_dir, "arrays.h5")
            hf = h5py.File(filepath, 'r')
            """
            self.indexes = np.array(hf.get('indexes'), dtype=np.uint32)
            self.token_ids = np.array(hf.get('token_ids'), dtype=np.uint16)
            self.lengths = np.array(hf.get('lengths'), dtype=np.uint16)
            """
            self.indexes = np.array(hf.get('indexes'))
            self.token_ids = np.array(hf.get('token_ids'), dtype=object)
            self.lengths = np.array(hf.get('lengths'))
            hf.close()
