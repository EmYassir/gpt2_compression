import os
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utilities.utils import logger


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6, reduction="sum"):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class ProbCrossEntropy(nn.Module):
    def __init__(self, reduction = 'sum'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        if self.reduction == 'mean':
            return -(y_true * y_pred).sum(dim=-1).mean()
        else:
            return -(y_true * y_pred).sum(dim=-1).sum()

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
        log_probs = -F.log_softmax(logits, dim=-1)
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
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
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
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        log_probs = -F.log_softmax(logits, dim=-1)
        nll_loss = model_output["loss"]
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        # Take the mean over the samples
        vocab_size = log_probs.shape[-1]
        smoothed_loss = smoothed_loss.mean() / vocab_size
        #logger.info(f'############## NLL LOSS = {nll_loss}')
        #logger.info(f'############## SMOOTHED LOSS = {smoothed_loss}')
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss



class MiniLMLoss:
    def __init__(self, alpha_att = 1., alpha_val = 1., temperature = 1.):
        self.alpha_att = alpha_att
        self.alpha_val = alpha_val
        self.temperature = temperature
        self.loss_fct = nn.KLDivLoss(reduction="batchmean")

    def __call__(self, student_outputs, teacher_outputs):
        loss = 0.

        if self.alpha_att != 0:
            t_sa = torch.stack(teacher_outputs.attentions)[-1]
            s_sa = torch.stack(student_outputs.attentions)[-1]
            att_loss = self.loss_fct(
                F.log_softmax(s_sa.view(-1, s_sa.size(-1)) / self.temperature, dim=-1, dtype=torch.float32), 
                F.softmax(t_sa.view(-1, t_sa.size(-1))  / self.temperature, dim=-1, dtype=torch.float32)
                )
            #logger.info(f'############## ATTENTION LOSS = {self.alpha_att * att_loss}')
            loss += self.alpha_att * att_loss # Log_softmax

        if self.alpha_val != 0: 
            v_loss = 0.
            # The past_key_vales are of size of (2, #layers, #batch, #heads, #seq_len, #hidden_dim/#heads) 
            s_v = torch.stack(tuple(map(torch.stack, zip(*list(student_outputs.past_key_values)))))
            t_v = torch.stack(tuple(map(torch.stack, zip(*list(teacher_outputs.past_key_values)))))
            s_v = student_outputs.past_key_values[-1][-1] # (#batch, #heads, #seq_len, #hidden_dim/#heads) 
            t_v = teacher_outputs.past_key_values[-1][-1] # (#batch, #heads, #seq_len, #hidden_dim/#heads) 
            
            s_denom, t_denom = (s_v.size(-1) * s_v.size(1)) ** 0.5, (t_v.size(-1) * t_v.size(1)) ** 0.5
            tt_v = torch.matmul(t_v, t_v.transpose(2, 3)) / s_denom # (batch_size , attn_heads , seq_len , seq_len)
            ss_v = torch.matmul(s_v, s_v.transpose(2, 3))  / t_denom # (batch_size , attn_heads , seq_len , seq_len)
            #v_loss += self.loss_fct(
            #    F.log_softmax(ss_v.view(-1, s_v.size(-1)) / self.temperature, dim=-1, dtype=torch.float32), 
            #    F.softmax(tt_v.view(-1, t_v.size(-1))  / self.temperature, dim=-1, dtype=torch.float32)
            #    )
            v_loss += self.loss_fct(
                F.log_softmax(ss_v.view(-1, ss_v.size(-1)) / self.temperature, dim=-1, dtype=torch.float32), 
                F.softmax(tt_v.view(-1, tt_v.size(-1))  / self.temperature, dim=-1, dtype=torch.float32)
                )
            loss += self.alpha_val * v_loss
            #logger.info(f'############## VALUE LOSS = {self.alpha_val * v_loss}')

        return loss


class ProjPKDLoss:
    def __init__(
        self, 
        s_input, 
        t_input, 
        teacher_layers, 
        output_dim = 0, 
        temperature = 1., 
        std_range = 0.02, 
        alpha = 1.,
        beta = 1.,
        classification = False,
        device = None
        ):
        self.student_input = s_input
        self.teacher_input = t_input
        self.output_dim = output_dim
        self.classification = classification
        if device is None:
            device = torch.device('cpu')
        self.teacher_layers = (torch.LongTensor(teacher_layers) + 1).to(device)

        self.temperature = temperature
        self.pt_loss_fct = nn.MSELoss(reduction="mean")
        self.ds_loss_fct = ProbCrossEntropy(reduction="mean")
        self.alpha = alpha
        self.beta = beta
        if self.output_dim > 0:
            self.s_random_proj = nn.Linear(self.student_input, self.output_dim).to(device)
            self.t_random_proj = nn.Linear(self.teacher_input, self.output_dim).to(device)
            for module in [self.s_random_proj, self.t_random_proj]:
                module.weight.data.normal_(mean=0.0, std=std_range)
                if module.bias is not None:
                    module.bias.data.zero_()    
                # Cancels automatic differentiation
                for p in module.parameters():
                    p.requires_grad=False
    

    def __call__(self, student_outputs, teacher_outputs, attention_mask):
        pt_loss, ds_loss, loss = 0.0, 0.0, 0.0

        if self.beta != 0:
            # Stack hidden states
            t_hs = torch.stack(teacher_outputs.hidden_states) # (s_layers, bs, seq_length, dim)
            s_hs = torch.stack(student_outputs.hidden_states) # (t_layers, bs, seq_length, dim)

            # PT Loss
            for i, index in enumerate(self.teacher_layers):
                s_elem = s_hs[i + 1, :, :, :] # (bs, seq_length, dim)
                t_elem = t_hs[index, :, :, :] # (bs, seq_length, dim)

                # Project
                if self.output_dim > 0:
                    s_elem = self.s_random_proj(s_elem)
                    t_elem = self.t_random_proj(t_elem)

                # Reshape and normalize
                s_elem = F.normalize(s_elem.view(-1, s_elem.size(-1)), p=2, dim=-1) # (bs * seq_length, dim)
                t_elem = F.normalize(t_elem.view(-1, t_elem.size(-1)), p=2, dim=-1) # (bs * seq_length, dim)

                # PT loss
                pt_loss += self.pt_loss_fct(s_elem, t_elem)
                
            #logger.info(f'############## PT LOSS = {pt_loss}')
            loss += self.beta * pt_loss

        if self.alpha != 0:
            s_logits, t_logits = student_outputs.logits, teacher_outputs.logits
            if self.classification == False:
                mask = (attention_mask.unsqueeze(-1).expand_as(s_logits) != 0 ) # (bs, seq_length, voc_size)
                s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
                t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
                t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
            else:
                s_logits_slct = s_logits.view(-1, s_logits.size(-1))
                t_logits_slct = t_logits.view(-1, t_logits.size(-1))

            # CE loss
            ds_loss += self.ds_loss_fct(
                F.log_softmax(s_logits_slct / self.temperature, dim=-1), 
                F.softmax(t_logits_slct / self.temperature, dim=-1)
                )
            loss +=  self.alpha * ds_loss
            #logger.info(f'############## DS LOSS = {ds_loss}')
        
        return loss
