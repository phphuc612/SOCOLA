import argparse
import os
from typing import Optional, get_args

from configs.literal import SUPPORTED_MODELS


def RATIO_TYPE(arg) -> float:
    """Type function for argparse - a float within some predefined bounds"""
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError(
            "Argument must be < " + str(1) + "and > " + str(0)
        )
    return f


def VALID_PATH_TYPE(arg) -> str:
    """Type function for argparse - a valid path"""
    if not os.path.exists(arg):
        raise argparse.ArgumentTypeError("Path does not exist")
    return arg.__str__()


class DefaultParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(DefaultParser, self).__init__(*args, **kwargs)
        self.add_default_args()
        self.add_specific_args()

    def add_default_args(self):
        self.add_argument("--data-dir", metavar="DIR", help="path to dataset")

        # Default model config
        supported_models = get_args(SUPPORTED_MODELS)
        self.add_argument(
            "--arch",
            metavar="ARCH",
            default="resnet50",
            choices=supported_models,
            help="model architecture: "
            + " | ".join(supported_models)
            + " (default: resnet50)",
        )

        self.add_argument(
            "--hidden-dim",
            default=128,
            type=int,
            help="feature dimension (default: 128)",
        )

        self.add_argument(
            "--mlp",
            action="store_true",
            help="use mlp head",
        )

        # Default dataloader config
        self.add_argument(
            "--subset",
            default=1,
            type=RATIO_TYPE,
            help="subset of data to use",
        )

        self.add_argument(
            "--aug-plus",
            action="store_true",
            help="use moco v2 data augmentation",
        )

        self.add_argument(
            "--save-dir",
            required=True,
            type=str,
            metavar="DIR",
            help="save directory",
        )

        self.add_argument(
            "--workers",
            default=32,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 32)",
        )

        self.add_argument(
            "--epochs",
            default=50,
            type=int,
            metavar="N",
            help="number of total epochs to run",
        )

        self.add_argument(
            "-b",
            "--batch-size",
            default=256,
            type=int,
            metavar="N",
            help="mini-batch size (default: 256), this is the total "
            "batch size of all GPUs on the current node when "
            "using Data Parallel or Distributed Data Parallel",
        )

        self.add_argument(
            "--lr",
            "--learning-rate",
            default=0.03,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )

        self.add_argument(
            "--schedule",
            default=[],
            nargs="*",
            type=int,
            help="learning rate schedule (when to drop lr by 10x)",
        )

        self.add_argument(
            "--wd",
            "--weight-decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )

        self.add_argument(
            "--momentum",
            default=0.9,
            type=RATIO_TYPE,
            metavar="M",
            help="momentum of SGD solver [default: 0.9]",
        )

        self.add_argument(
            "--cos",
            action="store_true",
            help="use cosine lr schedule",
        )

        # Default logging config
        self.add_argument(
            "-p",
            "--print-freq",
            default=10,
            type=int,
            metavar="N",
            help="print frequency (default: 10)",
        )

        self.add_argument(
            "--run-name",
            type=str,
            required=True,
            help="run name for wandb",
        )

        self.add_argument(
            "--debug",
            action="store_true",
            help="debug mode",
        )

        # Default other actions
        self.add_argument(
            "--resume",
            type=VALID_PATH_TYPE,
            metavar="PATH",
            help="path to latest checkpoint (default: none)",
        )

        self.add_argument(
            "--eval-only",
            type=VALID_PATH_TYPE,
            help="path to checkpoint for evaluation only",
        )

        # Default GPU config
        self.add_argument(
            "--world-size",
            default=-1,
            type=int,
            help="number of nodes for distributed training",
        )

        self.add_argument(
            "--rank",
            default=-1,
            type=int,
            help="node rank for distributed training",
        )

        self.add_argument(
            "--dist-url",
            default="tcp://224.66.41.62:23456",
            type=str,
            help="url used to set up distributed training",
        )

        self.add_argument(
            "--dist-backend",
            default="nccl",
            type=str,
            help="distributed backend",
        )

        self.add_argument(
            "--seed",
            default=None,
            type=int,
            help="seed for initializing training.",
        )

        self.add_argument(
            "--gpu",
            default=None,
            type=int,
            help="GPU id to use.",
        )

        self.add_argument(
            "--multiprocessing-distributed",
            action="store_true",
            help="Use multi-processing distributed training to launch "
            "N processes per node, which has N GPUs. This is the "
            "fastest way to use PyTorch for either single node or "
            "multi node data parallel training",
        )

    def add_specific_args(self):
        pass


class MoCoParser(DefaultParser):
    def add_specific_args(self):
        self.add_argument(
            "--moco-queue-size",
            default=65536,
            type=int,
            help="queue size; number of negative keys (default: 65536)",
        )

        self.add_argument(
            "--moco-momentum",
            default=0.999,
            type=RATIO_TYPE,
            help="moco momentum of updating key encoder (default: 0.999)",
        )

        self.add_argument(
            "--moco-temperature",
            default=0.07,
            type=float,
            help="softmax temperature (default: 0.07)",
        )


class SocolaParser(DefaultParser):
    def add_specific_args(self):
        self.add_argument(
            "--socola-temperature",
            default=None,
            type=Optional[float],
            help="softmax temperature (default: 0.07)",
        )
