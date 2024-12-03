# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from sklearn import metrics
import math

def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()
    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    
    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)

class WandbLogger(object):

    def __init__(self, login, dir=None, project=None, name=None, config={}):
        # -- check wandb support
        self.logger = False
        if login:
            import wandb
            wandb.login()
            self.logger = wandb.init(dir=dir, project=project, name=name, config=config)

        
    def log(self, data):
        if self.logger:
            self.logger.log(data)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if math.isnan(val): return
        
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats


def evaluate_scores(logits, labels, pos_label=1):
    """Evaluate different metric on logits, and labels.
    @param logits (tensor.FloatTensor of shape (batch_size, 2)) Model prediction. Each element correspond to [prob_class = 0, prob_class=1]
    @param labels (tensor.FloatTensor of shape (batch_size)) Ground truth prediction
    @return dict of score"""
    
    probs = logits.softmax(dim=1).detach()
    
    y_score = probs[:, 1]
    y_pred = probs.argmax(dim=1)
    y_true = labels.long().detach()

    y_true = y_true.cpu().numpy()
    y_score = y_score.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    return {
        "auc": metrics.auc(fpr, tpr),
        "recall": metrics.recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    }
        

def auroc(logits, labels):
    """Evaluate the auroc metric on logits, and labels.
    @param logits (tensor.FloatTensor of shape (batch_size, 2)) Model prediction. Each element correspond to [prob_class = 0, prob_class=1]
    @param labels (tensor.FloatTensor of shape (batch_size)) Ground truth prediction
    @return auc score"""
    
    probs = logits.softmax(dim=1).detach()
    
    y_score = probs[:, 1]
    y_true = labels.long().detach()

    fpr, tpr, thresholds = metrics.roc_curve(y_true.cpu().numpy(), y_score.cpu().numpy(), pos_label=1)
    return metrics.auc(fpr, tpr)



def recall(logits, labels):
    """Evaluate the recall metric on logits, and labels.
    @param logits (tensor.FloatTensor of shape (batch_size, 2)) Model prediction. Each element correspond to [prob_class = 0, prob_class=1]
    @param labels (tensor.FloatTensor of shape (batch_size)) Ground truth prediction
    @return auc score"""
    
    probs = logits.softmax(dim=1).detach()
    
    y_score = probs[:, 1]
    y_true = labels.long().detach()

    return metrics.recall_score(y_true.cpu().numpy(), y_score.cpu().numpy(), pos_label=1, zero_division=0)
    #return metrics.auc(fpr, tpr)