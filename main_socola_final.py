#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import logging
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm

import numpy as np
import psutil
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.datasets as datasets
# import torchvision.models as models
# import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from PIL import Image

import moco.socola_final
import moco.loader
import utils
import wandb
from dataset.coco import COCODataset

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)
run = None
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " +
    " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.03,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=20,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--dim", default=128, type=int, help="feature dimension (default: 128)"
)
parser.add_argument(
    "--T", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)
parser.add_argument(
    "--sub-batch-size", default=32, type=int, help="sub batch size (default: 32)"
)
# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument("--mlp_dim", default=4096, type=int, help="mlp head dimension")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true",
                    help="use cosine lr schedule")
parser.add_argument("--subset", default=1, type=float,
                    help="subset of data to use")

parser.add_argument("--save-dir", default=".", type=str, help="save directory")

parser.add_argument("--eval-only", default="",
                    help="path to checkpoint for evaluation only")

parser.add_argument("--run-name", default="", help="run name for wandb")
parser.add_argument("--debug", action="store_true", help="debug mode")
def main():
    args = parser.parse_args()
    runid = None
    if args.resume or args.eval_only:
        if not args.resume:
            path = args.eval_only
        else:
            path = args.resume
        parent_dir = os.path.dirname(path)
        print("parend dir", parent_dir)
        with open(os.path.join(parent_dir, "runid.txt"), "r") as f:
            runid = f.read().strip()
    global run
    if not args.debug:
        run = wandb.init(
            project="SOCOLA",
            group="final-pretrain" if not args.eval_only else "knn",
            name=args.run_name,
            config={
                "arch": args.arch,
                "temp": args.T,
                "dim": args.dim,
                "sub_batch_size": args.sub_batch_size,
                "lr": args.lr,
                "cos": args.cos,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "workers": args.workers,
                "seed": args.seed,
                "world_size": args.world_size,
                "rank": args.rank,
                "dist_url": args.dist_url,
                "dist_backend": args.dist_backend,
                "gpu": args.gpu,
                "multiprocessing_distributed": args.multiprocessing_distributed,
                "schedule": args.schedule,
                "aug_plus": args.aug_plus,
                "mlp": args.mlp,
                "subset": args.subset,
            },
            id=runid,
            resume="allow",
        )
        wandb.define_metric("epochs")
        wandb.define_metric("epochs/*", step_metric="epochs")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith("vit"):
        model = moco.socola_final.SOCOLA_ViT(
            models.__dict__[args.arch],
            args.sub_batch_size,
            args.dim,
            args.mlp_dim,
            args.T,
        )
    else:
        model = moco.socola_final.SOCOLA_Resnet(
            models.__dict__[args.arch],
            args.sub_batch_size,
            args.dim,
            args.mlp_dim,
            args.T,
        )
    print(model)
    criterion = None
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        # define loss function (criterion) and optimizer
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        model = model.to(device)

    # TODO: switch to ADAMW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.eval_only:
        if os.path.isfile(args.eval_only):
            print("=> loading checkpoint '{}'".format(args.eval_only))
            if args.gpu is None:
                checkpoint = torch.load(args.eval_only)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.eval_only, map_location=loc)
            model.load_state_dict(checkpoint["state_dict"])
            args.start_epoch = checkpoint["epoch"] - 1
            args.epochs = args.start_epoch + 1
            print("=> loaded checkpoint '{}'".format(args.eval_only))
    cudnn.benchmark = True

    # Data loading code
    datadir = args.data
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    train_dataset = datasets.ImageFolder(
        os.path.join(datadir, "ILSVRC/imagenet"),
        moco.loader.TwoCropsTransform(
            transforms.Compose(augmentation)
        ),
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(datadir, "ILSVRC/imagenet_val"),
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    class_index = None
    if args.subset < 1:
        class_index = train_dataset.classes
        subset = int(len(train_dataset) * args.subset)
        indices = torch.randperm(len(train_dataset))[:subset]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    # train_dataset = COCODataset(
    #     data_dir=os.path.join(datadir, "coco"),
    #     transform=moco.loader.TwoCropsTransform(
    #         transforms.Compose(augmentation)
    #     ),
    #     annotation_dir=os.path.join(datadir, "coco/annotations"),
    #     split="val",
    # )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None
    print("distributed", args.distributed)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    if args.eval_only:
        print("evaluating")
        img_encoder = model.encoder_img
        # eval_output = evaluate(val_loader, model, args.start_epoch, device, args)
        # eval_output = {f"epochs/eval-{k}": v for k, v in eval_output.items()}
        knn = kNN(args, args.start_epoch, img_encoder, train_loader, val_loader, 200, 0.07, class_index=class_index)
        knn = {f"epochs/knn-{k}": v for k, v in knn.items()}
        if not args.debug:
            wandb.log({**knn, "epochs": args.start_epoch})
        return 
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

    
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_output = train(train_loader, model,
                                optimizer, epoch, device, args)
        train_output = {f"epochs/train-{k}": v for k,
                        v in train_output.items()}
        wandb_log_output = {"epochs": epoch, **train_output}
        if not args.debug:
            wandb.log(wandb_log_output)
        # eval_output = evaluate(val_loader, model, epoch, device, args)
        # eval_output = {f"epochs/eval-{k}": v for k, v in eval_output.items()}
        # wandb_log_output.update(eval_output)

        # img_encoder = model.encoder_img

        # # kNN monitor
        # kNN_output = kNN(args, epoch, img_encoder, train_loader, val_loader, 200, 0.07)
        # knn_output = {f"epochs/knn-{k}": v for k, v in kNN_output.items()}
        # wandb_log_output.update(knn_output)

        if not args.eval_only and not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            pwd = os.getcwd()
            directory = os.path.join(pwd, args.save_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)
                if not args.resume:
                    with open(os.path.join(directory, "runid.txt"), "w") as f:
                        f.write(run.id)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=os.path.join(
                    directory, "checkpoint_{:04d}.pth.tar".format(epoch)),
            )


def train(
    data_loader,
    model,
    optimizer,
    epoch,
    device,
    args,
):
    # train
    if not args.debug:
        wandb.watch(model, log="all", log_freq=20)
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "temp", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "avg_loss", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = args.print_freq
    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    print(len(data_loader))

    for iteration_cnt, (imgs, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, logger, header)
    ):
        # logger.info(f"ram: {int(np.round(psutil.virtual_memory()[3] / (1000. **3))) }")  # total physical memory in Bytes
        optimizer.zero_grad(set_to_none=True)

        query_img = imgs[0].to(device)
        key_img = imgs[1].to(device)
        avg_loss, logits, labels = model(query_img, key_img, device)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        optimizer.step()

        metric_logger.update(avg_loss=avg_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temp=model.T.item())
        metric_logger.update(acc1=acc1[0].item())
        metric_logger.update(acc5=acc5[0].item())
        if iteration_cnt % print_freq == 0 and not args.debug:
            wandb.log({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "temp": model.T.item(),
                "acc1": acc1[0].item(),
                "acc5": acc5[0].item(),
            }, step=int(iteration_cnt / len(data_loader) * 100) + epoch * 100)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")

    # Encode the whole training set and store them into the memory bank
    # with torch.no_grad():

    return {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }


def evaluate(
    data_loader,
    model,
    epoch,
    device,
    args,
):
    # eval
    print("evaluate step")
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "avg_loss", utils.SmoothedValue(window_size=50, fmt="{value:.4f}")
    )

    header = "Eval Epoch: [{}]".format(epoch)
    print_freq = args.print_freq

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    print(len(data_loader))

    with torch.no_grad():
        for _, (imgs, _) in enumerate(data_loader):
            query_img = imgs[0].to(device)
            key_img = imgs[1].to(device)
            avg_loss, logits, labels = model(query_img, key_img, device)

            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

            metric_logger.update(avg_loss=avg_loss)
            metric_logger.update(acc1=acc1[0].item())
            metric_logger.update(acc5=acc5[0].item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger.global_avg()}")
    return {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }



def kNN(args, epoch, model, trainloader, testloader, K, sigma, class_index=None):
    """
    model: a vision model, like resnet
    """

    # TODO: lemniscate -> trainFeature bank, after complete one training epoch, encode the whole training set 
    # and store the features in the bank, then use the bank to do kNN

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "model_time", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "cls_time", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )
    metric_logger.add_meter(
        "top1", utils.SmoothedValue(window_size=50, fmt="{value:.2f}")
    )
    metric_logger.add_meter(
        "top5", utils.SmoothedValue(window_size=50, fmt="{value:.2f}")
    )

    header = "Test Epoch: [{}]".format(epoch)

    total = 0
    testsize = testloader.dataset.__len__()

    train_features = []
    train_labels = []
    C = None
    if hasattr(trainloader.dataset, 'classes'):
        C = len(trainloader.dataset.classes)
    elif class_index is not None:
        C = len(class_index)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(trainloader, desc="Encoding training set")):
            chosen_image = random.choice(images)         
            chosen_image = chosen_image.to(device)
            labels = labels.to(device)
            
            # random choose 1 from 2 image in images
            feature = model(chosen_image)
            train_features.append(feature.detach())
            train_labels.append(labels)

    train_features = torch.concat(train_features, dim=0)
    train_labels = torch.concat(train_labels, dim=0)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    # normalize the features?
    train_features = F.normalize(train_features, dim=1)
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).to(device)
        for batch_idx, (inputs, targets) in enumerate(
            metric_logger.log_every(testloader, args.print_freq, logger, header)
        ):
            end = time.time()
            targets = targets.to(device)
            inputs = inputs.to(device)
            batchSize = inputs.size(0)
            features = model(inputs)
            
            # normalize the features?
            features = F.normalize(features, dim=1)

            metric_logger.update(model_time=time.time() - end)
            # model_time.update(time.time() - end)
            end = time.time()

            # dist = torch.mm(features, train_features) # D x C, N x C -> D x N
            dist = features @ train_features.t() # bs x D, D x N -> bs x N

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))
            metric_logger.update(cls_time=time.time() - end)
            # cls_time.update(time.time() - end)
            
            top1 = top1 + correct.narrow(1,0,1).sum().item()
            top5 = top5 + correct.narrow(1,0,5).sum().item()

            total += targets.size(0)
            metric_logger.update(top1=top1 * 100. / total)
            metric_logger.update(top5=top5 * 100. / total)
    return {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
    }

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # N x maxk -> maxk x N
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # maxk x N

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
