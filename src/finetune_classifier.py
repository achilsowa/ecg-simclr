# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel



from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    WandbLogger,
    gpu_timer,
    grad_logger,
    evaluate_scores,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.mhiecg import make_mhiecg

from src.helper import (
    load_checkpoint,
    init_model,
    init_pretrained_model,
    init_opt)
from src.models.finetune_classifier import simclr_finetuned_classifier

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    encoder_path = args['meta']['pretrained_model_path']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    data_train_path = args['data']['data_train_path']
    data_test_path = args['data']['data_test_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    label_key = args['data']['label_key']
    # --


    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    dump = os.path.join(folder, 'params-finetuning.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'accuracy'),
                           ('%.5f', 'auc'),
                           ('%.5f', 'recall'),
                           # ('%.5f', 'mask-A'),
                           # ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))
    
    # -- make wandb_logger
    wandb_logger = WandbLogger(
        login=args['logging']['wandb_login'],
        dir=os.path.join(folder),
        project=args['meta']['project'], 
        name=args['meta']['name'], 
        config=args)
    
    # -- init model
    if encoder_path:
        encoder, projection = init_pretrained_model(
            device=device,
            encoder_path=encoder_path,
            get_model_fn=simclr_finetuned_classifier)
    else:
        encoder, projection, _ = init_model(
            device=device, 
            get_model_fn=lambda: simclr_finetuned_classifier()+(None, )
        )

    
    # -- init data-loaders/samplers    
    _, train_loader, train_sampler = make_mhiecg(
            batch_size=batch_size,
            label_keys=[label_key],
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            data_path=data_train_path,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(train_loader)
    _, test_loader, _ = make_mhiecg(
            batch_size=batch_size,
            label_keys=[label_key],
            pin_mem=pin_mem,
            training=False,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            data_path=data_test_path,
            copy_data=copy_data,
            drop_last=True)
    

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=torch.nn.Sequential(), # No optimization for the encoder
        projection=projection,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    # if torch.cuda.is_available():
    #    encoder = DistributedDataParallel(encoder, static_graph=True)
    #    projection = DistributedDataParallel(projection, static_graph=True)

    # for p in target_encoder.parameters():
    #     p.requires_grad = False

    # -- momentum schedule
    # momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
    #                       for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, projection, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            projection=projection,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            # next(momentum_scheduler)
            # mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'projection': projection.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        auc_meter = AverageMeter()
        recall_meter = AverageMeter()
        for itr, batch in enumerate(train_loader):
            projection.train()
            def load_ecgs():
                # -- unsupervised ecg
                ecgs, labels = batch["ecg"], batch[label_key]
                ecgs = ecgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                return ecgs, labels

            ecgs, labels = load_ecgs()
        
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                logits = projection(encoder(ecgs))
                loss = F.cross_entropy(logits, labels)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == labels).float().mean()
                score = evaluate_scores(logits, labels)
                
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                grad_stats = grad_logger(projection.named_parameters())
                optimizer.zero_grad()

                return (float(loss), float(accuracy), score, _new_lr, _new_wd, grad_stats)
            
            (loss, accuracy, score, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
            accuracy_meter.update(accuracy)
            auc_meter.update(score["auc"])
            recall_meter.update(score["recall"])

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, accuracy, score["auc"], score["recall"], etime)
                # wandb_logger.log({
                #     "epoch": epoch + 1, 
                #     "itr": itr, 
                #     "train-loss": loss, 
                #     "train-accuracy": accuracy, 
                #     "train-auc": auc,
                #     "time": etime
                # })

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f, accuracy: %.3f, auc: %.3f, recall: %.3f '
                                # 'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   accuracy_meter.avg,
                                   auc_meter.avg,
                                   recall_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f, avg. accuracy %.3f, avg. auc %.3f, avg. recall %.3f' % 
                    (loss_meter.avg, accuracy_meter.avg, auc_meter.avg, recall_meter.avg))
        wandb_logger.log({
            "epoch": epoch + 1, 
            "valid-loss": loss_meter.avg, 
            "valid-auc": auc_meter.avg,
            "valid-recall": recall_meter.avg,
            "valid-accuracy": accuracy_meter.avg, 
        })
        save_checkpoint(epoch+1)
        
        # -- TESTING
        def run_test():
            projection.eval()
            auc_meter = AverageMeter()
            accuracy_meter = AverageMeter()
            recall_meter = AverageMeter()
            loss_meter = AverageMeter()
            for itr, batch in enumerate(test_loader):

                def load_ecgs():
                    ecgs, labels = batch["ecg"], batch[label_key]
                    ecgs = ecgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    return ecgs, labels
                    
                ecgs, labels = load_ecgs()
                logits = projection(encoder(ecgs))
                loss = F.cross_entropy(logits, labels)
                predictions = logits.argmax(dim=1)
                accuracy = (predictions == labels).float().mean()
                score = evaluate_scores(logits, labels)
                
                auc_meter.update(score["auc"])
                recall_meter.update(score["recall"])
                accuracy_meter.update(float(accuracy))
                loss_meter.update(float(loss))
                
                # wandb_logger.log({"test-iter": itr, "test-blue": blue_meter.val})
                logger.info('test [%d, %d], accuracy: %.3f, auc: %.3f, recall: %.3f [mem: %.2e] ' % 
                            (epoch+1, itr, accuracy, score["auc"], score["recall"], torch.cuda.max_memory_allocated() / 1024.**2,))


            wandb_logger.log({
                "test-auc": auc_meter.avg, 
                "test-accuracy": accuracy_meter.avg,  
                "test-loss": loss_meter.avg,  
                "test-recall": recall_meter.avg,
                "epoch": epoch+1
            })
            logger.info('avg. loss %.3f, avg. accuracy %.3f, avg. auc %.3f, avg. recall %.3f '
                        '[mem: %.2e] ' % (loss_meter.avg, accuracy_meter.avg, auc_meter.avg, recall_meter.avg,  
                                          torch.cuda.max_memory_allocated() / 1024.**2,))

        run_test()
    

    
                

if __name__ == "__main__":
    main()
