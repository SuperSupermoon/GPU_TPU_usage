import os
import time
from glob import glob
from tqdm import tqdm
from datetime import datetime
import numpy as np
import math
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dataset import get_ddp_loader as get_loader
from models import get_pretraining_model
from optimizers import get_optimizer
from config.arg import argument
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def mp_train(args):
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args), join=True)

def setup(args, rank, world_size, pid):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    os.environ["RANK"] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group(args.dist_backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print('| distributed init (rank {}): {}'.format(
        rank, args.dist_url), flush=True)
    dist.barrier()

class dataset(Dataset):
    def __init__(self, args, data_dir):
        if split == 'train':
            self.dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
        else:
            self.dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'valid'))
        self.args = args
        self.train = train
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return self.transform(img)
        

def train(gpu, ngpus_per_node, args):
    setup(args, gpu, ngpus_per_node, pid)
    device = torch.device(args.device)

    print(f"Running basic DDP on rank {args.gpu}.")
    batch_size = int(args.train_batch_size / args.world_size)

    train_datasets = dataset(args, '/home/data_storage/imagenet')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets)
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=batch_size, pin_memory=True, num_workers=8, sampler=train_sampler,  drop_last = True)

    model = get_pretraining_model(args)
    model.to(device)
    
    if utils.is_main_process():
        for param_tensor in model.state_dict():
            print("builded model ", param_tensor)

    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_groups = utils.get_params_groups(model)
    optimizer = get_optimizer(args, params_groups)

    global_progress = tqdm(range(0, args.train_epoch + 1), desc=f'Training')
    for epoch in global_progress:
        train_sampler.set_epoch(epoch)
        model.train()
        for itr, (img1,img2) in enumerate(train_loader):
            train_itr = len(data_loader) * (epoch - 1) + itr  # global training iteration
            loss = model(img1.to(device), img2.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dist.all_reduce(loss.cuda(), op=dist.ReduceOp.SUM)
            reduced_loss=loss.item()/float(args.world_size)
            torch.cuda.synchronize()
            print("training loss", reduced_loss)


if __name__ == '__main__':
    print(f'Pytorch version: {torch.__version__}')
    args = argument()
    
    if args.mp:
        mp_train(args)
    else:
        train(None, None, args)
