# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the HiLoss, MaskSoftTargetCrossEntropy, knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import random

class HiLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra hiloss.
    """
    def __init__(self, base_criterion: torch.nn.Module, sig_criterion: torch.nn.Module):
        super().__init__()
        self.base_criterion = base_criterion
        self.sig_criterion = sig_criterion
        
    def select_background(self, outputs_list_n, labels_list_n, much):
        values, indices = torch.max(labels_list_n, dim=1)
        true_or_false = (indices == labels_list_n.shape[1]-1)
        num_background = torch.sum(true_or_false)
        keep = labels_list_n[~true_or_false]
        keep_index = torch.LongTensor(range(0,len(labels_list_n)))[~true_or_false]
        background_index = torch.LongTensor(range(0,len(labels_list_n)))[true_or_false]
        pick_index = background_index[torch.LongTensor(random.sample(range(num_background), num_background//much))]
        pick = labels_list_n[pick_index]
        final_label = torch.cat([keep, pick])
        final_index = torch.cat([keep_index, pick_index])
        final_sample = outputs_list_n[final_index]
        
        return final_sample, final_label 

    def forward(self, outputs_list, labels_list):
        """
        Args:
            outputs_list: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels_list: the 12 hilabels for the base criterion
            masks_list: to filter out missing labels
        """
        
        outputs_list = list(outputs_list)
        #labels_list = list(labels_list)
        #outputs_list[8], labels_list[8] = self.select_background(outputs_list[8], labels_list[8], 20)
        #outputs_list[9], labels_list[9] = self.select_background(outputs_list[9], labels_list[9], 20)
        #outputs_list[10], labels_list[10] = self.select_background(outputs_list[10], labels_list[10], 20)
        #print(labels_list)
        #exit()
        
        sig_loss_0 = self.base_criterion(outputs_list[0][0], labels_list[0])
        sig_loss_1 = self.base_criterion(outputs_list[0][1], labels_list[1])
        sig_loss_2 = self.base_criterion(outputs_list[0][2], labels_list[2])
        sig_loss_3 = self.base_criterion(outputs_list[0][3], labels_list[3])
        sig_loss_4 = self.base_criterion(outputs_list[0][4], labels_list[4])
        sig_loss_5 = self.base_criterion(outputs_list[0][5], labels_list[5])
        sig_loss_6 = self.base_criterion(outputs_list[0][6], labels_list[6])
        sig_loss_7 = self.base_criterion(outputs_list[0][7], labels_list[7])
        sig_loss_8 = self.base_criterion(outputs_list[0][8], labels_list[8])
        sig_loss_9 = self.base_criterion(outputs_list[0][9], labels_list[9])
        sig_loss_10 = self.base_criterion(outputs_list[0][10], labels_list[10])
        
        base_loss_final = self.base_criterion(outputs_list[-1], labels_list[11])
        
        #weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.3, 0.5, 1] # change 1
        #weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 
        #weights = [0, 0, 0, 0, 0.1, 0.15, 0.15, 0.15, 0.15, 0.3, 0.5, 1]
        #weights = [0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0.3, 0.5, 1]
        weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 1, 1, 1]
        #weights = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        assert len(weights) == 12

        loss = sig_loss_0 * weights[0] + sig_loss_1 * weights[1] + sig_loss_2 * weights[2] + sig_loss_3 * weights[3] + \
               sig_loss_4 * weights[4] + sig_loss_5 * weights[5] + sig_loss_6 * weights[6] + sig_loss_7 * weights[7] + \
               sig_loss_8 * weights[8] + sig_loss_9 * weights[9] + sig_loss_10 * weights[10] + \
               base_loss_final * weights[11] 
        
        return loss, \
               sig_loss_0, sig_loss_1, sig_loss_2, sig_loss_3, sig_loss_4, sig_loss_5, \
               sig_loss_6, sig_loss_7, sig_loss_8, sig_loss_9, sig_loss_10, \
               base_loss_final
    

class SigSoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SigSoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * torch.log(x), dim=-1)
        return loss.mean()
    
    
    
class MaskSoftTargetCrossEntropy(nn.Module):
    """
    This module wraps a standard SoftTargetCrossEntropy and adds an extra mask to filter out meaningless logits.
    """
    def __init__(self):
        super(MaskSoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: B x nb_classes, target: B x nb_classes, mask: B with 0 or 1
        """
        x = x[mask == 1]
        target = target[mask == 1]
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
