#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simclr.helper.py: Helper function
Achille Sowa <achilsowa@gmail.com>
Apr 1 23, 2024
"""

import torch
import logging
import sys

import torch


from src.utils.schedulers import (
    WarmupCosineSchedule,
    CosineWDSchedule)
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def one_hot(indexes, size, dim=-1):
    """
    Return the one_hot representation of indexes

    @param indexes(tensor.LongTensor of any shape)
    @param size(int) size each 1-of-k encoding
    @param dim(int or array of int) dimension to work on. By default, the last
    @return onehot(tensor.LongTensor of shape indexes.shape + (size))
    """
    E = torch.zeros(indexes.shape + (size, ), device=indexes.device)
    E.scatter_(dim, indexes.unsqueeze(dim).long(), 1)
    return E
        


def load_checkpoint(
    device,
    r_path,
    encoder,
    projection,
    opt=None,
    scaler=None,
):
    """Load the model. opt and scaler are optional, in which case they are not loaded"""
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        
        # -- loading encoder
        pretrained_dict = checkpoint['encoder']
        msg = encoder.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

        # -- loading projection
        pretrained_dict = checkpoint['projection']
        msg = projection.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained projection from epoch {epoch} with msg: {msg}')


        if opt is not None and scaler is not None:
            # -- loading optimizer
            opt.load_state_dict(checkpoint['opt'])
            if scaler is not None:
                scaler.load_state_dict(checkpoint['scaler'])
            logger.info(f'loaded optimizers from epoch {epoch}')
            logger.info(f'read-path: {r_path}')
            del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return encoder, projection, opt, scaler, epoch


def init_model(
    device,
    get_model_fn,
):
    encoder, projection, model = get_model_fn()

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)

    for m in projection.modules():
        init_weights(m)

    encoder.to(device)
    projection.to(device)
    logger.info(encoder)
    return encoder, projection, model


def init_opt(
    encoder,
    projection,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in projection.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in projection.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_pretrained_model(device, encoder_path, get_model_fn, update_pretrained_weight=False):
    encoder, projection = get_model_fn()

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    # -- initializing model weights
    for m in projection.modules():
        init_weights(m)

    # -- loading encoder
    checkpoint = torch.load(encoder_path, map_location=torch.device('cpu'))    
    pretrained_dict = checkpoint['encoder']
    msg = encoder.load_state_dict(pretrained_dict)
    for p in encoder.parameters():
        p.requires_grad = update_pretrained_weight
    logger.info(f'loaded pretrained encoder from path {encoder_path} with msg: {msg}')

    encoder.to(device)
    projection.to(device)
    logger.info(encoder)
    return encoder, projection

