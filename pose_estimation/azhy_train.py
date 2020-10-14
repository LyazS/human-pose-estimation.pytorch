# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models

from core.evaluate import accuracy
from tqdm import tqdm
from utils.vis import save_debug_images
device = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        required=False,
        default="experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml",
        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    model=models.pose_resnet.get_pose_net2().to(device)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=True).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [90,120], 0.1)

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = dataset.mpii(
        config, '/workspace/cpfs-data/datasets/mpii/', "train", True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    valid_dataset = dataset.mpii(
        config, '/workspace/cpfs-data/datasets/mpii/', "valid", False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    best_perf = 0.0
    best_model = False
    for epoch in range(140):
        for i, (input, target, target_weight,
                meta) in tqdm(enumerate(train_loader),
                              total=train_loader.__len__(),
                              leave=False,
                              desc=str(epoch)):
            # compute output
            output = model(input.to(device))
            target = target.to(device)
            target_weight = target_weight.to(device)

            loss = criterion(output, target, target_weight)

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 300 == 0:
                _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                 target.detach().cpu().numpy())
                prefix = '{}_{}_{}'.format(os.path.join("output_dir", 'train'),
                                           epoch, i)
                save_debug_images(config, input, meta, target, pred * 4,
                                  output, prefix)
        lr_scheduler.step()
        torch.save(model.state_dict(), 'final_state.pth.tar')
        print("Saved!")


if __name__ == '__main__':
    main()
