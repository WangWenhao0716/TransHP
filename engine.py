# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import numpy as np
import pandas as pd

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss, HiLoss
import utils

import gc

def one_hot(x_1, x_2, num_classes, on_value_1, on_value_2, off_value=0.1, device='cuda'):
    x_1 = x_1.long().view(-1, 1)
    x_2 = x_2.long().view(-1, 1)
    #print(x_1 == x_2)
    return torch.full(
        (x_1.size()[0], num_classes), \
        off_value/num_classes, \
        device=device).scatter_\
        (1, x_1, on_value_1 - off_value/2 + off_value/num_classes).scatter_\
        (1, x_2, on_value_2 - off_value/2 + off_value/num_classes)*(x_1 != x_2) + \
        torch.full(
        (x_1.size()[0], num_classes), \
        off_value/num_classes, \
        device=device).scatter_\
        (1, x_1, on_value_1 + on_value_2 - off_value + off_value/num_classes)*(x_1 == x_2)

def train_one_epoch(model: torch.nn.Module, criterion: HiLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    idx_to_class_dict = np.load('idx_to_class_dict.npy', allow_pickle=True).item()
    classes = np.array(list(idx_to_class_dict.values()))
    #df_hi_label = pd.read_csv('hi_label.csv')
    df_hi_label_idx = pd.read_csv('hi_label_idx.csv', index_col = 0)

    nb_classes_list = [2, 2, 2, 4, 9, 16, 25, 49, 90, 170, 406, 1000]
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        select_classes = classes[targets]
        targets_ls = list(targets.numpy())
        targets_list = df_hi_label_idx.loc[targets_ls].T.values.tolist()[::-1] + [targets_ls] 
        '''
        for t in range(len(targets_list)):
            print("The %d layer"%t)
            print(targets_list[t])
        '''
        #masks_list = np.int64(np.array(targets_list) != -1)#.tolist()
        
        
        #targets_list = list(targets_list*(targets_list != -1)) # after getting masks, clean -1 to avoid cls error occuring.
        
        samples = samples.to(device, non_blocking=True)
        
        assert len(targets_list) == 12
        for i in range(len(targets_list)):
            #targets_list[i] = targets_list[i].to(device, non_blocking=True)
            #print(i, targets_list[i])
            for j in range(len(targets_list[i])):
                if targets_list[i][j] == -1:
                    targets_list[i][j] = nb_classes_list[i]
            #print(i, targets_list[i])
        
        #exit()
        targets_list = torch.Tensor(targets_list).type(torch.int64).to(device, non_blocking=True)
        
        
        with torch.cuda.amp.autocast():
            outputs_list = model(samples, targets_list)
            loss, loss_0, loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, \
            loss_7, loss_8, loss_9, loss_10, loss_11 \
            = criterion(outputs_list, targets_list)
            
        loss_value = loss.item()
        loss_value_0 = loss_0.item()
        loss_value_1 = loss_1.item()
        loss_value_2 = loss_2.item()
        loss_value_3 = loss_3.item()
        loss_value_4 = loss_4.item()
        loss_value_5 = loss_5.item()
        loss_value_6 = loss_6.item()
        loss_value_7 = loss_7.item()
        loss_value_8 = loss_8.item()
        loss_value_9 = loss_9.item()
        loss_value_10 = loss_10.item()
        loss_value_11 = loss_11.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        # Wait for update
        
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_0=loss_value_0)
        metric_logger.update(loss_1=loss_value_1)
        metric_logger.update(loss_2=loss_value_2)
        metric_logger.update(loss_3=loss_value_3)
        metric_logger.update(loss_4=loss_value_4)
        metric_logger.update(loss_5=loss_value_5)
        metric_logger.update(loss_6=loss_value_6)
        metric_logger.update(loss_7=loss_value_7)
        metric_logger.update(loss_8=loss_value_8)
        metric_logger.update(loss_9=loss_value_9)
        metric_logger.update(loss_10=loss_value_10)
        metric_logger.update(loss_11=loss_value_11)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, '_')
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
