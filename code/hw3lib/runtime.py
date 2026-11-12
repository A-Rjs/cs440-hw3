'Set CPU threads and random seeds.'
import os,random
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch
from torch import nn

def configure(seed=440):
    torch.set_num_threads(1)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
