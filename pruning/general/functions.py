import torch.nn as nn
import torch
import time
import numpy as np

def get_time(device):
    if "cuda" in device:
        ctime = torch.cuda.Event(enable_timing=True)
        ctime.record()
        return ctime
    else:
        return time.time()

def get_time_diff(device, end, start):
    if "cuda" in device:
        torch.cuda.synchronize()
        return (start.elapsed_time(end))*1E-3
    else:
        return end - start

def f_beta(a, b, beta):
    # Harmonic mean, where a is considered beta times more important than b
    assert a >= 0 and a <= 1
    assert b >= 0 and b <= 1
    return (1+beta**2) * a * b / ((b*beta**2) + a)

def get_rng_state(args):
    # Save previous state
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    cuda_state = None
    if args.device == "cuda":
        cuda_state = torch.cuda.get_rng_state()
    return torch_state, np_state, cuda_state

def set_rng_state(args, torch_state, np_state, cuda_state):
    # Load previous state
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)
    if args.device == "cuda":
       torch.cuda.set_rng_state(cuda_state)

def set_library_defaults(args):
    if args.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)
    # Set seeds
    # random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class _Linearize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)