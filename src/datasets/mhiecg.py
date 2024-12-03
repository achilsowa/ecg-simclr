# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

import numpy as np

from logging import getLogger

import torch
import pandas as pd

_GLOBAL_SEED = 0
logger = getLogger()

def make_mhiecg(
    batch_size,
    label_keys=[],
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    data_path=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None

):
    data_path = data_path or os.path.join(root_path, 'data/md-validated-train-500.csv')
    dataset = EcgDataset(src_file=data_path, label_keys=label_keys)
    logger.info('Ecg dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('Ecg data loader created')

    return dataset, data_loader, dist_sampler


class EcgDataset(torch.utils.data.Dataset):
    def __init__(self, src_file, label_keys=[]) -> None:
        super().__init__()
        self.data = pd.read_csv(src_file)
        self.label_keys = label_keys
    
    def __getitem__(self, index):
        try: 
            obj =  {
                "ecg": torch.tensor(np.load(self.data.loc[index, 'npy_path']), dtype=torch.float32).squeeze(-1).t(),
            }
            for key in self.label_keys:
                obj[key] = self.data.loc[index, key]

            return obj
        
        except Exception as e:
            print(f'An error occured at: {index}/{self.data.shape[0]}/{self.data.loc[index]}:  {e}')
            return self.__getitem__(index-1)
        
    def __len__(self):
        return len(self.data)



