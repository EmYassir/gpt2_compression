# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import pandas as pd
from tqdm import tqdm


from utilities.utils import logger
from utilities.grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from utilities.lm_seqs_dataset_modified import LmSeqsDataset


class Evaluator:
    def __init__(
        self, params: dict, dataset: LmSeqsDataset, student: nn.Module, teacher: nn.Module
    ):
        logger.info("#### Initializing Evaluator")
        self.params = params
        self.output_dir = params.output_dir
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16
        self.student = student
        self.teacher = teacher
        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.block_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)

        self.temperature = params.temperature
        assert self.temperature > 0.0

        self.alpha_ce_student = params.alpha_ce_student
        self.alpha_ce_teacher = params.alpha_ce_teacher
        self.alpha_div = params.alpha_div


        logger.info("Using CLM loss for LM step.")
        self.kl_loss_fct = nn.KLDivLoss(reduction='none')
        self.ce_loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

        logger.info("--- Initializing model optimizer")

        if self.fp16:
            self.student = self.student.half()
            self.teacher = self.teacher.half()

        if self.multi_gpu:
            if self.fp16:
                from apex.parallel import DistributedDataParallel
                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(self.student)
            else:
                from torch.nn.parallel import DistributedDataParallel
                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.student = DistributedDataParallel(
                    self.student,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )

        self.is_master = params.is_master


    def prepare_batch(self, batch):
        """
        Prepare the batch: from the token_ids and the lengths, compute the attention mask and the labels for CLM.

        Input:
        ------
            batch: `Tuple`
                token_ids: `torch.tensor(bs, seq_length)` - The token ids for each of the sequence. It is padded.
                lengths: `torch.tensor(bs)` - The lengths of each of the sequences in the batch.

        Output:
        -------
            token_ids: `torch.tensor(bs, seq_length)` - The token ids after the modifications for CLM.
            attn_mask: `torch.tensor(bs, seq_length)` - The attention mask for the self-attention.
            clm_labels: `torch.tensor(bs, seq_length)` - The causal language modeling labels. There is a -100 where there is nothing to predict.
        """
        indexes, token_ids, lengths = batch
        indexes, token_ids, lengths = self.round_batch(indexes=indexes, x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0) == indexes.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size
        return indexes, token_ids, attn_mask, clm_labels, lengths

    def round_batch(self, indexes: torch.tensor, x: torch.tensor, lengths: torch.tensor):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding, so that each dimension is a multiple of 8.

        Input:
        ------
            x: `torch.tensor(bs, seq_length)` - The token ids.
            lengths: `torch.tensor(bs, seq_length)` - The lengths of each of the sequence in the batch.

        Output:
        -------
            x:  `torch.tensor(new_bs, new_seq_length)` - The updated token ids.
            lengths: `torch.tensor(new_bs, new_seq_length)` - The updated lengths.
        """
        if not self.fp16 or len(lengths) < 8:
            return indexes, x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            indexes = indexes[idx]
            lengths = lengths[idx]
            slen = lengths.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            #pad_id = self.params.special_tok_ids["pad_token"]
            pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return indexes, x, lengths

    def evaluate_samples(self):
        
        if self.is_master:
            logger.info("Starting Evaluation")
        
        # No gradient updates here, we simply assess samples 
        self.student.eval()
        self.teacher.eval()

        # We save everything in dictionary to be converted to a dataframe, later on
        final_scores = {'LBL_INDEX':[],  'CE_STUDENT':[], 'CE_TEACHER':[], 'KL_DIV':[], 'LENGTHS':[]}


        if self.multi_gpu:
            torch.distributed.barrier()

        iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
        total_samples = 0
        
        for batch in iter_bar:
            if self.params.n_gpu > 0:
                batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
            indexes, token_ids, attn_mask, lm_labels, lengths = self.prepare_batch(batch=batch)
            s_ppl, t_ppl, kl_div = self.compute_metrics(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels)
            _, added_elements = self.update_records(indexes, s_ppl, t_ppl, kl_div, lengths, final_scores)
            iter_bar.set_postfix({"Total samples processed": f"{total_samples}"})
            iter_bar.update()
            for _ in range(added_elements):
                total_samples += 1
                if (self.params.checkpoint_interval > 0) and (total_samples % self.params.checkpoint_interval == 0):
                    if self.is_master:
                        logger.info(f'Checkpointing after {total_samples} samples processed')
                    self.save_checkpoint(final_scores, f'samples_ckpt')
        
        iter_bar.close()
        output_file = 'samples'
        logger.info(f"Save very last checkpoint as `{output_file}_{self.params.local_rank}.csv`...")
        self.save_checkpoint(final_scores)
        logger.info("Training is finished")
    
    def update_records(self, indexes: torch.tensor, studen_ppls: torch.tensor, teacher_ppls: torch.tensor, kl_divs: torch.tensor, lengths: torch.tensor, dictionary = None):
        assert indexes.size() == studen_ppls.size() == teacher_ppls.size() == kl_divs.size() == lengths.size()
        if dictionary == None:
            dictionary = {'LBL_INDEX':[],  'CE_STUDENT':[], 'CE_TEACHER':[], 'KL_DIV':[], 'LENGTHS':[]}
        added_elements = 0
        for index, s_ppl, t_ppl, kl_div, length in zip(indexes, studen_ppls, teacher_ppls, kl_divs, lengths):
            # Updating scores 
            dictionary['LBL_INDEX'].append(int(index.item()))
            dictionary['CE_STUDENT'].append(float(s_ppl.item()))
            dictionary['CE_TEACHER'].append(float(t_ppl.item()))
            dictionary['KL_DIV'].append(float(kl_div.item()))
            dictionary['LENGTHS'].append(int(length.item()))
            added_elements += 1
        return dictionary, added_elements 



    def compute_metrics(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor):
        with torch.no_grad():
            output = self.student(input_ids=input_ids, attention_mask=None)
            s_logits = output.logits # (bs, seq_length, voc_size)
            output = self.teacher(input_ids=input_ids, attention_mask=None) 
            t_logits = output.logits # (bs, seq_length, voc_size)
            assert s_logits.size() == t_logits.size()
        mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        #s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits * mask 
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(0))  # (seq_length * voc_size, bs) modulo the 1s in mask
        #t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits * mask 
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(0))  # (seq_length * voc_size, bs) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        kl_div = (self.alpha_div * (
            self.kl_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            ) * (self.temperature) ** 2
        )).mean(0)

        # Cross entropy (student)
        shift_logits = s_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()
        """
        if self.is_master:
            logger.info(f'@@@@@@@@@@ shift_logits SIZE == {shift_logits.size()}')
            logger.info(f'@@@@@@@@@@ shift_labels SIZE == {shift_labels.size()}')
        """
        loss_ce = self.ce_loss_fct(shift_logits.view(-1, s_logits.size(-1)), shift_labels.view(-1))
        #s_ppl = (self.alpha_ce_student * torch.exp(loss_ce.view(s_logits.size(0), -1))).mean(-1)
        s_ppl = torch.exp((self.alpha_ce_student * loss_ce.view(s_logits.size(0), -1)).mean(-1))

        # Cross entropy (teacher)
        shift_logits = t_logits[..., :-1, :].contiguous()
        loss_ce = self.ce_loss_fct(shift_logits.view(-1, t_logits.size(-1)), shift_labels.view(-1))
        #t_ppl = (self.alpha_ce_teacher * torch.exp(loss_ce.view(self.params.batch_size, -1))).mean(-1)
        t_ppl = torch.exp((self.alpha_ce_teacher * loss_ce.view(t_logits.size(0), -1)).mean(-1))
        """
        if self.is_master:
            logger.info(f'########## STUDENT PPL SIZE == {s_ppl.size()}')
            logger.info(f'########## TEACHER PPL SIZE == {t_ppl.size()}')
            logger.info(f'########## KL DIV SIZE == {kl_div.size()}')
            logger.info(f'########## STUDENT PPL MEAN == {s_ppl.mean()}, STUDENT PPL SUM == {s_ppl.sum()}')
            logger.info(f'########## TEACHER PPL MEAN == {t_ppl.mean()}, TEACHER PPL SUM == {t_ppl.sum()}')
            logger.info(f'########## KL DIV MEAN == {kl_div.mean()}')
        if self.is_master:
            logger.info(f'########## STUDENT PPL MEAN == {s_ppl.mean()}')
            logger.info(f'########## TEACHER PPL MEAN == {t_ppl.mean()}')
            logger.info(f'########## KL DIV MEAN == {kl_div.mean()}')
        """
        return s_ppl, t_ppl, kl_div

  


    def save_checkpoint(self, dictionary, checkpoint_name: str = "samples_final"):
        """
        Save the current state. Only by the master process.
        """
        if dictionary is None:
            return
        logger.info(f'--> Converting the scores to a pandas dataframe...')
        df = pd.DataFrame.from_dict(dictionary)
        df.set_index('LBL_INDEX')
        output_file = os.path.join(self.output_dir, f'{checkpoint_name}_{self.params.local_rank}.csv')
        logger.info(f'--> Saving the scores to the file \'{output_file}\'')
        #df.to_csv(output_file, index_label='LBL_INDEX')
        df.to_csv(output_file)
            
