import argparse
from aux.loss import NNLoss, L0NormLoss
import torch
import numpy as np
import random
import os

loss_dict = {
    'nn_loss': NNLoss,
    'l0_norm_loss': L0NormLoss
}

def load_rng_state_from_file(args, address):
    assert os.path.isfile(address)
    # Load saved state
    saved_rng = torch.load(address)
    set_rng_state(args, saved_rng['torch'], saved_rng['numpy'], saved_rng['cuda'])

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
    torch.set_default_dtype(getattr(torch, args.dtype))
    # Set seeds
    # random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MainParser(argparse.ArgumentParser):

    def _modify_argument(self, which_arg, which_attr, new_v):
        # e.g. ("batch_size","default",100)
        for action in self._actions:
            if action.dest == which_arg:
                setattr(action, which_attr, new_v)
                return
        else:
            raise AssertionError('argument {} not found'.format(which_arg))

    def __init__(self):
        super().__init__()
        self.add_argument('--method', type=str, default='loorf_cos')
        self.add_argument('--loss_type', type=str, default='nn_loss')
        self.add_argument('--no_log', action="store_true")
        self.add_argument('--save', action="store_true", help="Whether to save and reload training information")
        self.add_argument('--duplicate_logs', action="store_true")
        self.add_argument('--out_id', type=str, default=None)
        self.add_argument('--out_dir', type=str, default="out")
        self.add_argument('--out_h5', type=str, default="df.h5")
        self.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
        self.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'])
        self.add_argument('--seed', type=int, default=1)
        self.add_argument('--epochs', type=int, default=300)
        self.add_argument('--val_interval', type=int, default=5)
        self.add_argument('--lr', type=float, default=1e-1)
        self.add_argument('--mask_init', type=float, default=0.5)
        self.add_argument('--visualize', action="store_true")
        self.add_argument('--lr_sch_pct', type=float, nargs='*', default=[],
                          help='Decrease learning rate at these percentages of training.')
        self.add_argument('--lr_sch_mul', type=float, nargs='*', default=[], help='Multipliers for LR schedule.')
        self.add_argument('--optim_str', type=str, default="rmsprop", choices=["adam","sgd","rmsprop"])
        self.add_argument('--log_final_mask', action="store_true")
        self.add_argument('--loss_rng', type=str, default="aux/rng.pt")

    def parse_args(self, *args, **kwargs):
        _args = super().parse_args(*args, **kwargs)
        _args.log = not _args.no_log
        set_library_defaults(_args)

        torch_state, np_state, cuda_state = get_rng_state(_args)
        load_rng_state_from_file(_args, _args.loss_rng)
        _args.loss_obj = loss_dict[_args.loss_type](_args)
        set_rng_state(_args, torch_state, np_state, cuda_state)

        _args.d = _args.loss_obj.d
        _args.lr_sch_epochs = [int(np.floor(v * _args.epochs)) for v in _args.lr_sch_pct]
        return _args