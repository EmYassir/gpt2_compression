""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
#import sys
#sys.path.append('/home/yassir/gpt2-ks/transformers_local')
import math
import os
import time
import psutil

# Integrations must be imported before ML frameworks:
from transformers_local.deepspeed import (
    deepspeed_init,
    is_deepspeed_zero3_enabled,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm
from typing import Optional

from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.file_utils import WEIGHTS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizer
from utilities.lm_seqs_dataset import LmSeqsDataset
from utilities.grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups



######## LOGGING ########

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Distiller:
    def __init__(
        self, params: dict, dataset: LmSeqsDataset, student: nn.Module, teacher: nn.Module, tokenizer: PreTrainedTokenizer
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.args = params # TODO to make cleaner
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16
        self.tokenizer = tokenizer

        # Experimental 
        self.deepspeed = params.deepspeed
        self.place_model_on_device = False if self.deepspeed else True

        self.model = student
        self.model_wrapped = student
        self.teacher = teacher

        self.student_config = student.config
        self.vocab_size = student.config.vocab_size

        
        if params.n_gpu <= 1:
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)
        
        if params.group_by_size:
            groups = create_lengths_groups(lengths=dataset.lengths, k=params.max_model_input_size)
            sampler = GroupedBatchSampler(sampler=sampler, group_ids=groups, batch_size=params.batch_size)
        else:
            sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)

        # DataLoaders creation:
        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=dataset.batch_sequences)

        self.temperature = params.temperature
        assert self.temperature > 0.0

        # Linear weights
        self.alpha_ce = params.alpha_ce
        self.alpha_clm = params.alpha_clm
        self.alpha_mse = params.alpha_mse
        self.alpha_cos = params.alpha_cos
        
        # Training settings
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_ce = 0
        self.last_loss_clm = 0
        if self.alpha_mse > 0.0:
            self.last_loss_mse = 0
        if self.alpha_cos > 0.0:
            self.last_loss_cos = 0
        self.last_log = 0

        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        if self.alpha_mse > 0.0:
            self.mse_loss_fct = nn.MSELoss(reduction="sum")
        if self.alpha_cos > 0.0:
            self.cosine_loss_fct = nn.CosineEmbeddingLoss(reduction="sum")

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )

        logger.info("--- Initializing model optimizer")
        assert params.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)
        num_train_optimization_steps = (
            int(self.num_steps_epoch / params.gradient_accumulation_steps * params.n_epoch) + 1
        )
        logger.info(f"--- Number of optimization steps: {num_train_optimization_steps}")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.model_wrapped.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.model_wrapped.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(params.adam_beta1, params.adam_beta2)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16 and not self.deepspeed:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.model_wrapped, self.optimizer = amp.initialize(
                self.model_wrapped, self.optimizer, opt_level=self.params.fp16_opt_level
            )

        if self.fp16:
            self.teacher = self.teacher.half()

        if self.multi_gpu and not self.deepspeed:# deepspeed manages its own fp16
            if self.fp16: 
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.model_wrapped = DistributedDataParallel(self.model_wrapped)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.model_wrapped = DistributedDataParallel(
                    self.model_wrapped,
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                )
            
        if self.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, 
                num_training_steps=num_train_optimization_steps, 
                resume_from_checkpoint=os.path.dirname(params.student_pretrained_weights)
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        self.is_master = params.is_master


    def prepare_batch_clm(self, batch):
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
        token_ids, lengths = batch
        token_ids, lengths = self.round_batch(x=token_ids, lengths=lengths)
        assert token_ids.size(0) == lengths.size(0)

        attn_mask = torch.arange(token_ids.size(1), dtype=torch.long, device=lengths.device) < lengths[:, None]
        clm_labels = token_ids.new(token_ids.size()).copy_(token_ids)
        clm_labels[~attn_mask] = -100  # previously `clm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility

        # sanity checks
        assert 0 <= token_ids.min() <= token_ids.max() < self.vocab_size

        return token_ids, attn_mask, clm_labels

    def round_batch(self, x: torch.tensor, lengths: torch.tensor):
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
            return x, lengths

        # number of sentences == 0 [8]
        bs1 = len(lengths)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
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
            pad_id = self.params.special_tok_ids["unk_token"]
            padding_tensor = torch.zeros(bs2, pad, dtype=torch.long, device=x.device).fill_(pad_id)
            x = torch.cat([x, padding_tensor], 1)
            assert x.size() == (bs2, ml2)

        assert x.size(0) % 8 == 0
        assert x.size(1) % 8 == 0
        return x, lengths

    '''
    def create_scheduler(self, num_training_steps: int):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            warmup_steps = (
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else math.ceil(num_training_steps * self.args.warmup_ratio)
            )

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
    '''
    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()

        model = self.wrap_model(self.model_wrapped)
        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model
        
        model.zero_grad()
        self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)
                token_ids, attn_mask, lm_labels = self.prepare_batch_clm(batch=batch)
                self.step(model, input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels)
                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        #if self.is_master:
        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        logger.info("Training is finished")

    def step(self, model: nn.Module, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        model.train()
        student_output = model(input_ids=input_ids, attention_mask=None)
        s_logits, s_hidden_states = student_output.logits, student_output.hidden_states # (bs, seq_length, voc_size)
        with torch.no_grad():
            teacher_output = self.teacher(input_ids=input_ids, attention_mask=None)
        t_logits, t_hidden_states = teacher_output.logits, teacher_output.hidden_states # (bs, seq_length, voc_size)
        assert s_logits.size() == t_logits.size()

        if self.params.restrict_ce_to_mask:
            mask = (lm_labels > -1).unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        else:
            mask = attention_mask.unsqueeze(-1).expand_as(s_logits)  # (bs, seq_length, voc_size)
        s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()

        loss_ce = (
            self.ce_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1),
                F.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        loss = self.alpha_ce * loss_ce

        if self.alpha_clm > 0.0:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss += self.alpha_clm * loss_clm
        
        if self.alpha_mse > 0.0:
            loss_mse = self.mse_loss_fct(s_logits_slct, t_logits_slct) / s_logits_slct.size(
                0
            )  # Reproducing batchmean reduction
            loss += self.alpha_mse * loss_mse
        if self.alpha_cos > 0.0:
            s_hidden_states = s_hidden_states[-1]  # (bs, seq_length, dim)
            t_hidden_states = t_hidden_states[-1]  # (bs, seq_length, dim)
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden_states)  # (bs, seq_length, dim)
            assert s_hidden_states.size() == t_hidden_states.size()
            dim = s_hidden_states.size(-1)

            s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
            s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
            t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
            t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)

            target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1.)  # (bs * seq_length,)
            ## YE: NEED TO CAST TO FLOAT32, otherwise error!
            with torch.cuda.amp.autocast():
                loss_cos = self.cosine_loss_fct(s_hidden_states_slct,  t_hidden_states_slct, target)
                loss += self.alpha_cos * loss_cos

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_ce = loss_ce.item()
        if self.alpha_clm > 0.0:
            self.last_loss_clm = loss_clm.item()
        if self.alpha_mse > 0.0:
            self.last_loss_mse = loss_mse.item()
        if self.alpha_cos > 0.0:
            self.last_loss_cos = loss_cos.item()

        self.optimize(model, loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, model, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        """
        # Check for NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        if self.multi_gpu:
            loss = loss.mean()
        if self.params.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.params.gradient_accumulation_steps

        if self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        elif self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
        if self.deepspeed:
            self.deepspeed.step()
        elif self.n_iter % self.params.gradient_accumulation_steps == 0:
            # deepspeed does its own clipping
            if self.fp16 :
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model_wrapped.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        model.zero_grad()
        self.iter()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()


    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        ## Automatically checks for Master
        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0
    
    def wrap_model(self, model):
        # already initialized its own DDP and AMP
        if self.deepspeed:
            return self.deepspeed
        else:
            return model

    def save(self, output_dir: Optional[str] = None, state_dict=None, checkpoint_name: str = "checkpoint.pth"):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.params.dump_path
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model_wrapped, PreTrainedModel):
            unwrapped_model = unwrap_model(self.model_wrapped)
            if isinstance(unwrapped_model, PreTrainedModel):
                if state_dict is None:
                    state_dict = unwrapped_model.state_dict()
                unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, checkpoint_name))
        else:
            self.model_wrapped.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Good practice: save your training arguments together with the trained model
        torch.save(self.params, os.path.join(output_dir, "training_args.bin"))
    
    def save_model(self, output_dir: Optional[str] = None, checkpoint_name: str = "checkpoint.pth"):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.params.dump_path

        state_dict = self.model_wrapped.state_dict()

        if self.is_master:
            self.save(output_dir, state_dict=state_dict, checkpoint_name=checkpoint_name)
        
        if self.deepspeed:
            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.is_master:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_fp16_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                self.deepspeed.save_fp16_model(output_dir, WEIGHTS_NAME)
            else:
                # All processes need to save their chunks
                self.deepspeed.save_checkpoint(output_dir)


    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master and not self.deepspeed:
            return
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.n_total_iter}"
        output_dir = os.path.join(self.dump_path, checkpoint_folder)
        self.save_model(output_dir, checkpoint_name)
    
