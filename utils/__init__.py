import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .logger import *
from .meter import *
from .parallel_utils import *
from .time import *


# set random seed
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
