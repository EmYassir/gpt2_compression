""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
#import sys
#sys.path.append('/home/yassir/gpt2-ks/transformers_local')
import math
import os
import time
import psutil

# .deepspeed must be imported before ML frameworks:
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
from tqdm import tqdm
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from transformers_local.modeling_utils import PreTrainedModel, unwrap_model
from transformers_local.file_utils import WEIGHTS_NAME
from transformers_local.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers_local.optimization import get_scheduler
from transformers_local import get_linear_schedule_with_warmup, default_data_collator

from utilities.utils import logger, move_to
from utilities.loss_objects import MiniLMLoss, ProjPKDLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


def upcast_state_dict(state_dict):
    t_cast = torch.float32
    n_state_dict = state_dict.copy()
    for k, v in state_dict.items():
        if v.dtype != t_cast:
            n_state_dict[k] = 



def require_teacher(args):
    return args.pkd or args.minilm


class PKD_Distiller:
    def __init__(
        self, params: dict, dataset: torch.utils.data.dataset.Dataset, student: nn.Module, teacher: nn.Module, tokenizer
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

        # DataLoaders creation:
        self.dataloader = DataLoader(dataset=dataset, shuffle=True, collate_fn=default_data_collator, batch_size=params.batch_size)
        self.temperature = params.temperature
        assert self.temperature > 0.0

        # Linear weights
        self.alpha_lm = params.alpha_lm
        self.alpha_att = params.alpha_att
        self.alpha_val = params.alpha_val
        self.alpha_pkd = params.alpha_pkd
        self.beta_pkd = params.beta_pkd
        
        # Training settings
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0

        # Logging
        self.last_loss = 0
        self.last_loss_lm = 0
        if self.params.minilm:
            self.last_loss_minilm = 0
        if self.params.pkd:
            self.last_loss_pkd = 0
        self.last_log = 0

        self.pkd_loss_fct = ProjPKDLoss(
            s_input = student.config.hidden_size,
            t_input = teacher.config.hidden_size, 
            teacher_layers = params.teacher_layers, 
            output_dim = params.pkd_output, 
            temperature = self.temperature,
            std_range = self.params.std_range, 
            alpha = self.alpha_pkd,
            beta = self.beta_pkd,
            device = f"cuda:{self.params.local_rank}"
        ) if params.pkd else None

        self.minilm_loss_fct = MiniLMLoss(
            alpha_att = self.alpha_att, 
            alpha_val = self.alpha_val, 
            temperature = self.temperature
        ) if params.minilm else None


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
            % sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.model.parameters()]))
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
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.params.fp16_opt_level
            )

        if self.fp16 and require_teacher(self.params):
            self.teacher = self.teacher.half()

        if self.multi_gpu and not self.deepspeed:
            if self.fp16:
                from apex.parallel import DistributedDataParallel

                logger.info("Using apex.parallel.DistributedDataParallel for distributed training.")
                self.model = DistributedDataParallel(self.model)
            else:
                from torch.nn.parallel import DistributedDataParallel

                logger.info("Using nn.parallel.DistributedDataParallel for distributed training.")
                self.model = DistributedDataParallel(
                    self.model,
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
        if self.is_master:
            logger.info("--- Initializing Tensorboard")
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)




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
        if require_teacher(self.params):
            self.teacher.eval()

        for _ in range(self.params.n_epoch):
            if self.is_master:
                logger.info(f"--- Starting epoch {self.epoch}/{self.params.n_epoch-1}")
            if self.multi_gpu:
                torch.distributed.barrier()

            iter_bar = tqdm(self.dataloader, desc="-Iter", disable=self.params.local_rank not in [-1, 0])
            for batch in iter_bar:
                if self.params.n_gpu > 0:
                    batch = move_to(batch, f"cuda:{self.params.local_rank}")
                
                token_ids, attn_mask, lm_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
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
        student_output = model(
            input_ids=input_ids, attention_mask=None, labels=lm_labels, output_attentions=True,
        )
        if require_teacher(self.params):
            with torch.no_grad():
                teacher_output = self.teacher(
                    input_ids=input_ids, attention_mask=None, output_attentions=True,
                )
            assert student_output.logits.size() == teacher_output.logits.size()
        
        # Init all losses
        loss_lm, loss_pkd, loss_minilm, loss = 0., 0., 0., 0.
        if self.alpha_lm != 0:
            loss_lm = student_output.loss
        if self.params.pkd:
            loss_pkd = self.pkd_loss_fct(student_output, teacher_output, attention_mask)
        if self.params.minilm:
            loss_minilm = self.minilm_loss_fct(student_output, teacher_output)

        #logger.info(f"################ LOSS_LM == {loss_lm}")
        #logger.info(f"################ LOSS_PKD == {loss_pkd}")
        #logger.info(f"################ LOSS_MINILM == {loss_minilm}")
        loss =  self.alpha_lm * loss_lm + loss_pkd + loss_minilm

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        #logger.info(f"################ LOSS == {loss.item()}")

        if self.alpha_lm != 0.0:
            self.last_loss_lm = loss_lm.item()
        if self.params.pkd:
            self.last_loss_pkd = loss_pkd.item()
        if self.params.minilm:
            self.last_loss_minilm = loss_minilm.item()

        self.optimize(model, loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, model, loss):
        """
        Normalization on the loss (gradient accumulation or distributed training), followed by
        backward pass on the loss, possibly followed by a parameter update (depending on the gradient accumulation).
        Also update the metrics for tensorboard.
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
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
        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        #if self.n_total_iter % self.params.checkpoint_interval == 0:
        #    self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.model.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        if self.alpha_lm > 0.0:
            self.tensorboard.add_scalar(
                tag="losses/loss_lm", scalar_value=self.last_loss_lm, global_step=self.n_total_iter
            )
        if self.params.pkd:
            self.tensorboard.add_scalar(
                tag="losses/loss_pkd", scalar_value=self.last_loss_pkd, global_step=self.n_total_iter
            )
        if self.params.minilm:
            self.tensorboard.add_scalar(
                tag="losses/loss_minilm", scalar_value=self.last_loss_minilm, global_step=self.n_total_iter
            )
        self.tensorboard.add_scalar(
            tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        )
        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(
            tag="global/speed", scalar_value=time.time() - self.last_log, global_step=self.n_total_iter
        )

    def end_epoch(self):
        """
        Finally arrived at the end of epoch (full pass on dataset).
        Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        ## Automatically checks for Master
        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
        if self.is_master:
            self.tensorboard.add_scalar(
                tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
            )

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

    def save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.params.dump_path
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            unwrapped_model = unwrap_model(self.model)
            if isinstance(unwrapped_model, PreTrainedModel):
                #if state_dict is None:
                #    state_dict = unwrapped_model.state_dict()
                state_dict = unwrapped_model.state_dict()
                logger.info(f"##### SAVING UNWRAPPED DICT {output_dir}")
                logger.info(f"Saving model checkpoint to {output_dir}")
                unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, 'pytorch_model'))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Good practice: save your training arguments together with the trained model
        torch.save(self.params, os.path.join(output_dir, "training_args.bin"))
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.params.dump_path


        state_dict = self.model.state_dict()

        if self.is_master and not self.deepspeed:
            self.save(output_dir, state_dict=state_dict)
        elif self.deepspeed:

            # this takes care of everything as long as we aren't under zero3
            if self.is_master:
                self.save(output_dir)

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


    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master and not self.deepspeed:
            return
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.n_total_iter}"
        output_dir = os.path.join(self.dump_path, checkpoint_folder)

        # We have to save the weights afterwards
        if self.is_master:
            self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        mdl_to_save = self.model.module if hasattr(self.model, "module") else self.model
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
