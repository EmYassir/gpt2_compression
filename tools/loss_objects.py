import collections
import gc
import inspect
import math
import os
import re
import shutil
import sys
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler


from transformers.utils import logging

class OriginalLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.
    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    def __init__(self, epsilon: float = 0.1, ignore_index: int = -100):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels.clamp_min_(0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        #print("\r\n#################################################\n")
        #print(f"####### NLL LOSS NUMERATOR = {nll_loss.sum()}`.\n")
        #print(f"####### NLL LOSS DENOMINATOR = {num_active_elements}`.\n")
        nll_loss = nll_loss.sum() / num_active_elements
        #print(f">>>>>>> NLL LOSS = {nll_loss}`.\n")


        #print(f"####### SMOOTHED LOSS NUMERATOR = {smoothed_loss.sum()}`.\n")
        #print(f"####### SMOOTHED LOSS DENOMINATOR = {num_active_elements * log_probs.shape[-1]}`.\n")   
        #print(f"####### NUM ACTIVE ELEMENTS = {num_active_elements}`.\n")
        #print(f"####### VOCAB SIZE = {log_probs.shape[-1]}`.\n")
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        #print(f">>>>>>> SMOOTHED LOSS = {smoothed_loss}`.\n")
        #print("#################################################\r\n")
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

class ModifiedOriginalLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.
    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    def __init__(self, epsilon: float = 0.1, ignore_index: int = -100):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        labels = labels[..., 1:].contiguous().view(-1)
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels.clamp_min_(0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        #print("\r\n#################################################\n")
        #print(f"####### NLL LOSS NUMERATOR = {nll_loss.sum()}`.\n")
        #print(f"####### NLL LOSS DENOMINATOR = {num_active_elements}`.\n")
        nll_loss = nll_loss.sum() / num_active_elements
        #print(f">>>>>>> NLL LOSS = {nll_loss}`.\n")


        #print(f"####### SMOOTHED LOSS NUMERATOR = {smoothed_loss.sum()}`.\n")
        #print(f"####### SMOOTHED LOSS DENOMINATOR = {num_active_elements * log_probs.shape[-1]}`.\n")   
        #print(f"####### NUM ACTIVE ELEMENTS = {num_active_elements}`.\n")
        #print(f"####### VOCAB SIZE = {log_probs.shape[-1]}`.\n")
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        #print(f">>>>>>> SMOOTHED LOSS = {smoothed_loss}`.\n")
        #print("#################################################\r\n")
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

class CustomLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.
    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    def __init__(self, epsilon: float = 0.1, ignore_index: int = -100):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
    

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels.clamp_min_(0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        nll_loss = nll_loss.sum()
        smoothed_loss = smoothed_loss.sum() / (log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

