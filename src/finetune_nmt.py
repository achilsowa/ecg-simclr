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
import math
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
    auroc,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.mhiecg import make_mhiecg

from src.helper import (
    load_checkpoint,
    init_model,
    init_pretrained_model,
    init_opt)
from src.models.finetune_nmt import simclr_finetuned_nmt
from src.utils.nmt import read_target_corpus, decode

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
    vocab_path = args['data']['vocab_path']
    tgt_model_path = args['data']['tgt_model_path']
    tgt_vocab_path = args['data']['tgt_vocab_path']
    test_output_path = args['data']['test_output_path']
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
    clip_grad = args['optimization']['clip_grad']

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
                           ('%.5f', 'ppl'),
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
    encoder, projection = init_pretrained_model(
        device=device,
        encoder_path=encoder_path,
        get_model_fn=lambda: simclr_finetuned_nmt(vocab_path))
    
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
        projection.train()
        
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        ppl_meter = AverageMeter()
        for itr, batch in enumerate(train_loader):

            def load_ecgs():
                # -- unsupervised ecg
                ecgs, labels = batch["ecg"], batch[label_key]
                ecgs = ecgs.to(device, non_blocking=True)
                return ecgs, read_target_corpus(labels, tgt_model_path)

            ecgs, labels = load_ecgs()
        
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --
                
                embeddings = encoder(ecgs).permute(0, 2, 1)
                batch_loss = -projection(embeddings, labels).sum()

                loss = batch_loss / len(labels)
                report_tgt_words = sum(len(s[1:]) for s in labels) # Omitting leading <s>
                ppl = math.exp(loss.item() / report_tgt_words)
                
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(projection.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(projection.parameters(), clip_grad)
                    optimizer.step()

                grad_stats = grad_logger(projection.named_parameters())
                optimizer.zero_grad()

                return (float(loss), float(ppl), _new_lr, _new_wd, grad_stats)
            
            (loss, ppl, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
            ppl_meter.update(ppl)
            

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, ppl, etime)
                # wandb_logger.log({
                #     "epoch": epoch + 1, 
                #     "itr": itr, 
                #     "train-loss": loss, 
                #     "train-ppl": ppl, 
                #     "time": etime
                # })

                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f, ppl: %.3f '
                                # 'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   ppl_meter.avg,
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
        logger.info('avg. loss %.3f, avg. ppl %.3f' % (loss_meter.avg, ppl_meter.avg,))
        wandb_logger.log({
            "epoch": epoch + 1, 
            "train-loss": loss_meter.avg, 
            "train-ppl": ppl_meter.avg,
            "time": etime
        })
        save_checkpoint(epoch+1)


        # -- TESTING
        def run_test():
            projection.eval()
            ppl_meter = AverageMeter()
            blue_meter = AverageMeter()
            for itr, batch in enumerate(test_loader):

                def load_ecgs():
                    ecgs, labels = batch["ecg"], batch[label_key]
                    ecgs = ecgs.to(device, non_blocking=True)
                    return ecgs, read_target_corpus(labels, tgt_model_path)

                ecgs, labels = load_ecgs()
                embeddings = encoder(ecgs).permute(0, 2, 1)
                score = decode(projection, embeddings, labels, epoch+1, test_output_path)
                blue_meter.update(score)

                # wandb_logger.log({"test-iter": itr, "test-blue": blue_meter.val})
                logger.info('[%d, %d], blue: %.3f ' % (epoch+1, itr, blue_meter.val))


            wandb_logger.log({"test-blue": blue_meter.avg, "epoch": epoch+1})
            logger.info('avg. blue: %.3f ' % (blue_meter.avg,))

        run_test()
    

if __name__ == "__main__":
    main()
