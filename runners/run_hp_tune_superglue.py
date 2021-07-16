# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on SuperGLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import time
from typing import Dict, Callable, List, Optional, Tuple, Union, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining

from utilities.superglue_compute_metrics import superglue_compute_metrics as compute_metrics
from utilities.hp_search import (
    modified_compute_objective, 
    hp_search_setup, 
    tune_save_checkpoint,
    ray_hp_space_default,
    ray_hp_space_pkd_minilm,
    report_to_hp_search,
    run_hp_search_ray,
)
from transformers_local import superglue_convert_examples_to_features as convert_examples_to_features
from transformers_local import superglue_output_modes as output_modes
from transformers_local import superglue_processors as processors
from transformers_local import superglue_tasks_metrics as task_metrics
from transformers_local import superglue_tasks_num_spans as task_spans
from transformers_local.modeling_utils import PreTrainedModel
from transformers_local.tokenization_utils_base import PreTrainedTokenizerBase
from transformers_local.trainer_utils import EvalPrediction
from transformers_local.trainer_utils import (
    BestRun,
    EvalPrediction,
    set_seed,
    PREFIX_CHECKPOINT_DIR
)

from transformers_local import (  # AlbertForSequenceClassification,; AlbertTokenizer,; DistilBertForSequenceClassification,; DistilBertTokenizer,; FlaubertForSequenceClassification,; FlaubertTokenizer,; XLMForSequenceClassification,; XLMRobertaForSequenceClassification,; XLMRobertaTokenizer,; XLMTokenizer,; XLNetForSequenceClassification,; XLNetTokenizer,
    WEIGHTS_NAME,
    AdamW,
    PretrainedConfig,
    GPT2Config,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    GPT2ForSpanClassification,
    get_linear_schedule_with_warmup,
)
from utilities.loss_objects import MiniLMLoss, ProjPKDLoss

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

#from hanging_threads import start_monitoring
#start_monitoring(seconds_frozen=360, test_interval=360)

logger = logging.getLogger(__name__)

# ALL_MODELS = sum(
#     (
#         tuple(conf.pretrained_config_archive_map.keys())
#         for conf in (
#             GPT2Config,
#         )
#     ),
#     (),
# )

MODEL_CLASSES = {
    # "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    # "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    # "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    # "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    # "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
    "gpt2": (
        GPT2Config,
        GPT2Tokenizer,
        {"classification": GPT2ForSequenceClassification, "span_classification":  GPT2ForSpanClassification,},
    ),
}


TASK2FILENAME = {
    "boolq": "BoolQ.jsonl",
    "cb": "CB.jsonl",
    "copa": "COPA.jsonl",
    "multirc": "MultiRC.jsonl",
    "record": "ReCoRD.jsonl",
    "rte": "RTE.jsonl",
    "wic": "WiC.jsonl",
    "wsc": "WSC.jsonl",
}



class HP_Tuner:
    
    def __init__(
        self,
        args: Dict = None,
        model: Union[PreTrainedModel, torch.nn.Module] = None,
        teacher: Union[PreTrainedModel, torch.nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        self.args = args if args is not None else None
        self.model = model
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        self.loss_objects = {}

    def model_init(self, trial):
        # Prepare task
        task_name = self.args.task_name
        processor = processors[self.args.task_name]()
        output_mode = output_modes[self.args.task_name]
        label_list = processor.get_labels()
        num_labels = len(label_list)

        # Load pretrained model and tokenizer
        model_type = self.args.model_type.lower()
        config_class, _, model_classes = MODEL_CLASSES[model_type]
        model_class = model_classes[output_mode]
        config = config_class.from_pretrained(
            self.args.config_name if self.args.config_name else self.args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        config.pad_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        if output_mode == "span_classification":
            config.num_spans = task_spans[task_name]
        self.model = model_class.from_pretrained(
            self.args.model_name_or_path,
            from_tf=bool(".ckpt" in self.args.model_name_or_path),
            config=config,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
    
        # No need to re-Load teacher model
        """ 
        config = config_class.from_pretrained(
            self.args.teacher_name_or_path,
            num_labels=num_labels,
            finetuning_task=self.args.task_name,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        if self.args.output_mode == "span_classification":
            config.num_spans = task_spans[self.args.task_name]
        self.teacher = model_class.from_pretrained(
            self.args.teacher_name_or_path,
            from_tf=bool(".ckpt" in self.args.teacher_name_or_path),
            config=config,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        self.teacher.resize_token_embeddings(len(tokenizer))
        """
        ## Losses
        if self.args.pkd:
            self.loss_objects['pkd'] = ProjPKDLoss(
                s_input = self.model.config.hidden_size,
                t_input = self.teacher.config.hidden_size, 
                teacher_layers = self.args.teacher_layers, 
                output_dim = self.args.pkd_output, 
                temperature = trial['temperature'],
                std_range = self.args.std_range, 
                alpha = trial['alpha_pkd'],
                beta = trial['beta_pkd'],
                classification = True,
                device = self.args.device
            )
        if self.args.minilm:
            self.loss_objects['minilm'] = MiniLMLoss(
                alpha_att = trial['alpha_att'], 
                alpha_val = trial['alpha_val'], 
                temperature = trial['temperature']
            )
        return self.model

    def evaluate(self, split="dev", use_tqdm=True):
        results = {}
        if self.args.task_name == "record":
            eval_dataset, eval_answers = self.load_and_cache_examples(self.args.task_name, self.tokenizer, split=split)
        else:
            eval_dataset = self.load_and_cache_examples(self.args.task_name, self.tokenizer, split=split)
            eval_answers = None

        if not os.path.exists(self.args.output_dir) and self.args.local_rank in [-1, 0]:
            os.makedirs(self.args.output_dir)

        self.args.eval_batch_size = self.args.per_gpu_eval_batch_size * max(1, self.args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        model = self.model
        
        # multi-gpu eval
        if self.args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info(f"***** Running evaluation: on {self.args.task_name} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        ex_ids = None
        eval_dataloader = tqdm(eval_dataloader, desc="Evaluating") if use_tqdm else eval_dataloader
        # debug_idx = 0
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            guids = batch[-1]

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.args.output_mode == "span_classification":
                    inputs["spans"] = batch[4]
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
                ex_ids = [guids.detach().cpu().numpy()]
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                ex_ids.append(guids.detach().cpu().numpy())
            # debug_idx += 1
            # if debug_idx > 10:
            #    break

        ex_ids = np.concatenate(ex_ids, axis=0)
        eval_loss = eval_loss / nb_eval_steps
        if self.args.output_mode in ["classification", "span_classification"] and self.args.task_name not in ["record"]:
            preds = np.argmax(preds, axis=1)
        elif self.args.output_mode == "regression":
            preds = np.squeeze(preds)
        # logging.info(f"predictions: {preds}")
        if split != "test":
            # don't have access to test labels, so skip evaluating on them
            # NB(AW): forcing evaluation on ReCoRD on test (no labels) will error
            result = compute_metrics(self.args.task_name, preds, out_label_ids, guids=ex_ids, answers=eval_answers)
            results.update(result)
            results['eval_loss'] = eval_loss
            output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info(f"***** {split} results: {self.args.task_name} *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        return results, preds, ex_ids
    

    def train(self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Dict[str, Any] = None,
        **kwargs,):

        hp_search_setup(self.args, trial)
        self.model = self.model_init(trial)
        # Reinitializes optimizer and scheduler
        self.optimizer, self.lr_scheduler = None, None

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
            logger.info(f"Loading model from {resume_from_checkpoint}).")

            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self.model.load_state_dict(state_dict, strict=False)
        
        # Place models on device
        model = self.model.to(self.args.device)
        teacher = self.teacher.to(self.args.device)

        """ Train the model """
        # if args.local_rank in [-1, 0]:

        self.args.train_batch_size = self.args.per_gpu_train_batch_size * max(1, self.args.n_gpu)

        train_dataset = self.load_and_cache_examples(self.args.task_name, self.tokenizer, split="train")
        train_sampler = RandomSampler(train_dataset) if self.args.local_rank == -1 else DistributedSampler(self.train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:  # number of training steps = number of epochs * number of batches
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        num_warmup_steps = int(self.args.warmup_ratio * t_total)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(self.args.model_name_or_path, "scheduler.pt")
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(self.args.model_name_or_path, "scheduler.pt")))

        if self.args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)
            logger.info("Training with fp16.")

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.args.local_rank], output_device=self.args.local_rank, find_unused_parameters=True,
            )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", self.args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(self.args.model_name_or_path):
            # set global_step to global_step of last saved checkpoint from model path
            try:
                global_step = int(self.args.model_name_or_path.split("-")[-1].split("/")[0])
            except ValueError:
                global_step = 0
            epochs_trained = global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // self.args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        #train_iterator = trange(
        #    epochs_trained, int(self.args.num_train_epochs), desc="Epoch", disable=self.args.local_rank not in [-1, 0],
        # )
        train_iterator = range(epochs_trained, int(self.args.num_train_epochs))

        set_seed(self.args.seed)  # Added here for reproductibility
        best_val_metric = None
        for epoch_n in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_n}", disable=self.args.local_rank not in [-1, 0])
            epoch_iterator = train_dataloader
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.args.output_mode == "span_classification":
                    inputs["spans"] = batch[4]
                if self.args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                
                if self.args.pkd:
                    inputs["output_hidden_states"] = True
                
                if self.args.minilm:
                    inputs["output_attentions"] = True
                    inputs["use_cache"] = True
            
                # Teacher
                teacher_outputs = teacher(**inputs)
                
                # Student
                outputs = model(**inputs)
                #loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                ce_loss = self.args.alpha_ce * outputs.loss

                if self.args.pkd:
                    loss_pkd = self.loss_objects['pkd'](outputs, teacher_outputs, inputs['attention_mask'])
                
                if self.args.minilm:
                    loss_minilm = self.loss_objects['minilm'](outputs, teacher_outputs)

                loss =  ce_loss + loss_pkd + loss_minilm

                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                if self.args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                # epoch_iterator.set_description(f"Epoch {epoch_n} loss: {loss:.3f}")
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    results = None
                    logs = {}
                    if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        if (
                            self.args.local_rank == -1 and self.args.log_evaluate_during_training and results is None
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results, _, _ = self.evaluate(use_tqdm=False)
                            report_to_hp_search(self, trial, epoch_n, results)
                            for key, value in results.items():
                                eval_key = "eval_{}".format(key)
                                logs[eval_key] = value

                        loss_scalar = (tr_loss - logging_loss) / self.args.logging_steps
                        learning_rate_scalar = scheduler.get_last_lr()[0]
                        logs["learning_rate"] = learning_rate_scalar
                        logs["avg_loss_since_last_log"] = loss_scalar
                        logging_loss = tr_loss

                        # for key, value in logs.items():
                        #    tb_writer.add_scalar(key, value, global_step)
                        # print(json.dumps({**logs, **{"step": global_step}}))
                        logging.info(json.dumps({**logs, **{"step": global_step}}))

                    if (
                        self.args.local_rank in [-1, 0]
                        and self.args.eval_and_save_steps > 0
                        and global_step % self.args.eval_and_save_steps == 0
                    ):
                        # evaluate
                        results, _, _ = self.evaluate(use_tqdm=False)
                        report_to_hp_search(self, trial, epoch_n, results)
                        for key, value in results.items():
                            logs[f"eval_{key}"] = value
                        logger.info(json.dumps({**logs, **{"step": global_step}}))

                        # save
                        if self.args.save_only_best:
                            output_dirs = []
                        else:
                            output_dirs = [os.path.join(self.args.output_dir, f"checkpoint-{global_step}")]
                        curr_val_metric = results[task_metrics[self.args.task_name]]
                        if best_val_metric is None or curr_val_metric > best_val_metric:
                            # check if best model so far
                            logger.info("Congratulations, best model so far!")
                            output_dirs.append(os.path.join(self.args.output_dir, "checkpoint-best"))
                            best_val_metric = curr_val_metric

                        for output_dir in output_dirs:
                            # in each dir, save model, tokenizer, self.args, optimizer, scheduler
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir, exist_ok=True)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            logger.info("Saving model checkpoint to %s", output_dir)
                            model_to_save.save_pretrained(output_dir)
                            self.tokenizer.save_pretrained(output_dir)
                            torch.save(self.args, os.path.join(output_dir, "training_self.args.bin"))
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("\tSaved model checkpoint to %s", output_dir)

                if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and global_step >= self.args.max_steps:
                train_iterator.close()
                break

        # if self.args.local_rank in [-1, 0]:
        #    tb_writer.close()

        return global_step, tr_loss / global_step



    def load_and_cache_examples(self, task, tokenizer, split="train"):
        if self.args.local_rank not in [-1, 0] and split not in ["dev", "test"]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        processor = processors[task]()
        output_mode = output_modes[task]
        # Load data features from cache or dataset file
        cached_tensors_file = os.path.join(
            self.args.data_dir,
            "tensors_{}_{}_{}_{}".format(
                split, list(filter(None, self.args.model_name_or_path.split("/"))).pop(), str(self.args.max_seq_length), str(task),
            ),
        )
        if os.path.exists(cached_tensors_file) and not self.args.overwrite_cache:
            logger.info("Loading tensors from cached file %s", cached_tensors_file)
            start_time = time.time()
            dataset = torch.load(cached_tensors_file)
            logger.info("\tFinished loading tensors")
            logger.info(f"\tin {time.time() - start_time}s")

        else:
            # no cached tensors, process data from scratch
            logger.info("Creating features from dataset file at %s", self.args.data_dir)
            label_list = processor.get_labels()
            if split == "train":
                get_examples = processor.get_train_examples
            elif split == "dev":
                get_examples = processor.get_dev_examples
            elif split == "test":
                get_examples = processor.get_test_examples
            examples = get_examples(self.args.data_dir)
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=self.args.max_seq_length,
                output_mode=output_mode,
                pad_on_left=bool(self.args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.args.model_type in ["xlnet"] else 0,
            )
            logger.info("\tFinished creating features")
            if self.args.local_rank == 0 and split not in ["dev", "train"]:
                torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

            # Convert to Tensors and build dataset
            logger.info("Converting features into tensors")
            all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            if output_mode in ["classification", "span_classification"]:
                all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            elif output_mode == "regression":
                all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

            if output_mode in ["span_classification"]:
                # all_starts = torch.tensor([[s[0] for s in f.span_locs] for f in features], dtype=torch.long)
                # all_ends = torch.tensor([[s[1] for s in f.span_locs] for f in features], dtype=torch.long)
                all_spans = torch.tensor([f.span_locs for f in features])
                dataset = TensorDataset(
                    all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_spans, all_guids
                )
            else:
                dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guids)
            logger.info("\tFinished converting features into tensors")
            if self.args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_tensors_file)
                torch.save(dataset, cached_tensors_file)
                logger.info("\tFinished saving tensors")

        if self.args.task_name == "record" and split in ["dev", "test"]:
            if split=='dev':
                answers = processor.get_answers(self.args.data_dir, 'val')
            else:
                answers = processor.get_answers(self.args.data_dir, split)
            return dataset, answers
        else:
            return dataset
    
    def hyperparameter_search(
        self,
        hp_space: Optional[Dict[str, float]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union[str]] = "ray",
        hp_name: Optional[ str] = None,
        **kwargs,
    ) -> BestRun:
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = hp_space if hp_space is not None else ray_hp_space_pkd_minilm
        self.hp_name = hp_name
        self.compute_objective = modified_compute_objective if compute_objective is None else compute_objective

        best_run = run_hp_search_ray(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run




def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument(
        "--log_evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--adam_beta1", default=1e-8, type=float, help="Epsilon for Adam optimizer. Currently not used. "
    )
    parser.add_argument(
        "--adam_beta2", default=1e-8, type=float, help="Epsilon for Adam optimizer. Currently not used. "
    )
    parser.add_argument("--std_range", default=0.02, type=float, help="Random initialization range.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_steps as a float.")

    parser.add_argument("--log_energy_consumption", action="store_true", help="Whether to track energy consumption")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_and_save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best", action="store_true", help="Save only when hit best validation score.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--evaluate_test", action="store_true", help="Evaluate on the test splits.")
    parser.add_argument("--skip_evaluate_dev", action="store_true", help="Skip final evaluation on the dev splits.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--use_gpuid", type=int, default=-1, help="Use a specific GPU only")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    
    ## Distillation parameters
    parser.add_argument("--teacher_type", choices=["gpt2"], required=True, help="Teacher type (GPT2).")
    parser.add_argument("--teacher_name_or_path", type=str, required=True, help="The teacher type or weights.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Temperature for the softmax temperature.")

    parser.add_argument("--alpha_ce", default=0.2, type=float, help="Linear weight for the cross-entropy loss.")
    parser.add_argument("--minilm", action="store_true", help="Use of mini-lm loss")
    parser.add_argument("--alpha_att", default=0.2, type=float, help="Linear weight for the self attention loss (Mini-LM).")
    parser.add_argument("--alpha_val", default=0.2, type=float, help="Linear weight for the value-value loss (Mini-LM).")
    parser.add_argument("--pkd", action="store_true", help="Use of pkd loss")
    parser.add_argument("--alpha_pkd", default=0.2, type=float, help="Linear weight for the NLL loss (PKD).")
    parser.add_argument("--beta_pkd", default=0.2, type=float, help="Linear weight for the MSE loss (PKD).")
    parser.add_argument("--pkd_output", default=0, type=int, help="Output dimension for random projection.")
    parser.add_argument('--teacher_layers','--list', nargs='+', type=int, default=[0, 2, 4, 7, 9, 11])

    ## Other arguments
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        format="%(asctime)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Launch impact tracker
    """
    if args.log_energy_consumption:
        from experiment_impact_tracker.compute_tracker import ImpactTracker

        logger.info("Launching impact tracker...")
        tracker = ImpactTracker(args.output_dir)
        tracker.launch_impact_monitor()
    """
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    """
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    """

    # Setup CUDA, GPU & distributed training
    if args.use_gpuid > -1:
        device = args.use_gpuid
        args.n_gpu = 1
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args.seed)

    # Prepare task
    args.task_name = args.task_name.lower()
    assert args.task_name in processors, f"Task {args.task_name} not found!"
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Do all the stuff you want only first process to do
    # e.g. make sure only the first process will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, tokenizer_class, model_classes = MODEL_CLASSES[args.model_type]
    model_class = model_classes[args.output_mode]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.output_mode == "span_classification":
        config.num_spans = task_spans[args.task_name]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Load teacher model 
    args.teacher_type = args.teacher_type.lower()
    config_class, _, model_classes = MODEL_CLASSES[args.teacher_type]
    model_class = model_classes[args.output_mode]
    config = config_class.from_pretrained(
        args.teacher_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    if args.output_mode == "span_classification":
        config.num_spans = task_spans[args.task_name]
    teacher = model_class.from_pretrained(
        args.teacher_name_or_path,
        from_tf=bool(".ckpt" in args.teacher_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.resize_token_embeddings(len(tokenizer))
    teacher.resize_token_embeddings(len(tokenizer))

    #args.device = torch.device("cpu")
    #logger.info(f"$$$$$$$$$$$$ USING DEVICE {args.device}")
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    ## Hyperparameter tuning
    trainer = HP_Tuner(args=args, model=model, teacher=teacher, tokenizer=tokenizer)
    kwargs = dict(local_dir='/home/yassir/output/hp_search', resources_per_trial={"cpu": 4, "gpu": 1})
    best_run = trainer.hyperparameter_search(hp_space = ray_hp_space_pkd_minilm, compute_objective = None, n_trials = 10, direction = "maximize", backend = "ray", hp_name = None, **kwargs)
    logger.info(f"BEST RUN HYPERPARAMETERS == {best_run.hyperparameters}")
    filename = os.path.join(args.output_dir, 'best_hyperparameters.json')
    with open(filename, 'w') as fp:
        json.dump(fp, best_run.hyperparameters, indent = 4)

if __name__ == "__main__":
    main()
