""" The distiller to distil the student.
    Adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
#import sys
#sys.path.append('/home/yassir/gpt2-ks/transformers_local')
import math
import os
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

#from transformers import get_linear_schedule_with_warmup, default_data_collator
from transformers_local import get_linear_schedule_with_warmup, default_data_collator

from utilities.utils import logger, move_to
from utilities.loss_objects import MiniLMLoss, ProjPKDLoss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter



def require_teacher(args):
    return args.pkd or args.minilm


class PKD_Distiller:
    def __init__(
        self, params: dict, dataset: torch.utils.data.dataset.Dataset, student: nn.Module, teacher: nn.Module, tokenizer=None
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.multi_gpu = params.multi_gpu
        self.fp16 = params.fp16

        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer

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
            classification = False,
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
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.999)
        )

        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )

        if self.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            logger.info(f"Using fp16 training: {self.params.fp16_opt_level} level")
            self.student, self.optimizer = amp.initialize(
                self.student, self.optimizer, opt_level=self.params.fp16_opt_level
            )
            if require_teacher(self.params):
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
        if self.is_master:
            logger.info("--- Initializing Tensorboard")
            self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log", "train"))
            self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)
            self.tensorboard.add_text(tag="config/student", text_string=str(self.student_config), global_step=0)







    def train(self):
        """
        The real training loop.
        """
        if self.is_master:
            logger.info("Starting training")
        self.last_log = time.time()
        self.student.train()
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
                self.step(input_ids=token_ids, attention_mask=attn_mask, lm_labels=lm_labels)
                iter_bar.update()
                iter_bar.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_bar.close()

            if self.is_master:
                logger.info(f"--- Ending epoch {self.epoch}/{self.params.n_epoch-1}")
            self.end_epoch()

        if self.is_master:
            logger.info("Save very last checkpoint as `pytorch_model.bin`.")
            self.save_checkpoint(checkpoint_name="pytorch_model.bin")
            logger.info("Training is finished")

    def step(self, input_ids: torch.tensor, attention_mask: torch.tensor, lm_labels: torch.tensor):
        """
        One optimization step: forward of student AND teacher, backward on the loss (for gradient accumulation),
        and possibly a parameter update (depending on the gradient accumulation).

        Input:
        ------
        input_ids: `torch.tensor(bs, seq_length)` - The token ids.
        attention_mask: `torch.tensor(bs, seq_length)` - The attention mask for self attention.
        lm_labels: `torch.tensor(bs, seq_length)` - The language modeling labels (mlm labels for MLM and clm labels for CLM).
        """
        student_output = self.student(
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

        loss =  self.alpha_lm * loss_lm + loss_pkd + loss_minilm

        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()

        if self.alpha_lm != 0.0:
            self.last_loss_lm = loss_lm.item()
        if self.params.pkd:
            self.last_loss_pkd = loss_pkd.item()
        if self.params.minilm:
            self.last_loss_minilm = loss_minilm.item()

        self.optimize(loss)

        self.n_sequences_epoch += input_ids.size(0)

    def optimize(self, loss):
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
        if self.params.gradient_accumulation_steps > 1:
            loss = loss / self.params.gradient_accumulation_steps

        if self.fp16:
            from apex import amp

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.iter()
        if self.n_iter % self.params.gradient_accumulation_steps == 0:
            if self.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.params.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.params.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.params.log_interval == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.params.checkpoint_interval == 0:
            self.save_checkpoint()

    def log_tensorboard(self):
        """
        Log into tensorboard. Only by the master process.
        """
        if not self.is_master:
            return

        for param_name, param in self.student.named_parameters():
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

        if self.is_master:
            self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
            self.tensorboard.add_scalar(
                tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
            )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0

    def save_checkpoint(self, checkpoint_name: str = "checkpoint.pth"):
        """
        Save the current state. Only by the master process.
        """
        if not self.is_master:
            return
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        mdl_to_save.config.save_pretrained(self.dump_path)
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))
