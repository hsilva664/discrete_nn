import argparse
from config import Config
import numpy as np
import torch
from general.functions import set_library_defaults


# Just to make sure that non-standard types are not being passed to args unwillingly, as that variable gets recorded by pandas
def checktypes(f):
    def wrapper(*args, **kwargs):
        o_args = f(*args, **kwargs)
        recognized = [int, float, str, list, tuple, bool, np.float64]
        for k, v in vars(o_args).items():
            if not (type(v) in recognized or v is None):
                raise TypeError(f"Unrecognized type {type(v)}")
        return o_args
    return wrapper

class MainParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(allow_abbrev=False)
        all_methods, all_method_classes = zip(*Config.GET_METHOD_DICT().items())
        all_nn = Config.GET_BASE_NN_PARAMETERS().keys()
        all_datasets = Config.DATASET_PARAMS.keys()
        
        self.add_argument('--id', type=str, default=None)
        self.add_argument('--logdir', type=str, default='logs')
        self.add_argument('--seed', type=int, default=1)
        self.add_argument('--backbone_seed', type=int, default=None, help="Seed to use for backbone network (default = args.seed + 1000)")
        self.add_argument('--epochs', type=int, default=None)
        self.add_argument('--optim', type=str, default=None, choices=['adam', 'sgd', 'rmsprop'])
        self.add_argument('--lr', type=float, default=None)
        self.add_argument('--wd', type=float, default=None, help='Weight decay.')
        self.add_argument('--nesterov', default=None, action=argparse.BooleanOptionalAction, help='Whether to use nesterov (only works on SGD)')
        self.add_argument('--momentum', type=float, default=None, help='Momentum to use on optimizer (only works on SGD)')
        self.add_argument('--lr_sch_pct', type=float, nargs='*', default=None, help='Decrease learning rate at these percentages of training.')
        self.add_argument('--lr_sch_mul', type=float, nargs='*', default=None, help='Multipliers for LR schedule.')
        self.add_argument('--train_bs', type=int, default=None)
        self.add_argument('--device', default='cuda',choices=('cuda','cpu'))
        self.add_argument('--which_gpu', default=None, type=int)
        self.add_argument('--no_log', action='store_true', help="Whether to record data in a dataframe or not")
        self.add_argument('--use_double', action='store_true')
        self.add_argument('--method', default='base', choices=[a for a in all_methods] )
        self.add_argument('--nn', default='lenet', choices=all_nn )
        self.add_argument('--dataset', default='mnist',choices=[a for a in all_datasets])
        self.add_argument("--duplicate_logs", action="store_true", help="Allow duplicate logs on output file")
        self.add_argument('--out_h5', type=str, default="df.h5")
        self.add_argument('--n_workers', type=int, default=1)
        self.add_argument('--dont_train_backbone', action="store_true")
        self.add_argument('--eval_every', type=int, default=None, help="Epochs until eval")
        self.add_argument("--save", action="store_true", help="Save (and reload) the state of training."
                                                              "Meant to be used when runs will be interrupted halfway")
        self.add_argument("--keep_final_nn", action="store_true", help="When saving/reloading, do not delete the final nn")

    @checktypes
    def parse_args(self, *args, **kwargs):
        args, remaining = super().parse_known_args(*args, **kwargs)
        if args.id is None:
            args.id = args.method
        method_args = Config.GET_METHOD_DICT()[args.method].Parser().parse_args(remaining)
        a_dict = vars(method_args)
        a_dict.update(vars(args))
        args = argparse.Namespace(**a_dict)
        # Overwrite unspecified parameters with defaults from NN model
        dont_use = []
        if args.optim is not None:
            # Don't use default optimization parameters if the optimizer is custom
            dont_use = ['lr', 'wd', 'momentum', 'nesterov', ]
        BASE_NN_PARAMETERS = Config.GET_BASE_NN_PARAMETERS()
        for key, val in BASE_NN_PARAMETERS[args.nn].items():
            if key not in dont_use:
                if getattr(args, key, None) is None:
                    setattr(args, key, val)

        set_library_defaults(args)
        if args.which_gpu is not None:
            assert args.device == 'cuda'
            torch.cuda.set_device(args.which_gpu)

        args.train_backbone = not args.dont_train_backbone
        if not args.save:
            assert not args.keep_final_nn
        if args.backbone_seed is None:
            # Assumes seed + 1000 does not belong to the sweep
            args.backbone_seed = args.seed + 1000
        # LR schedule processing
        if args.train_backbone:
            assert len(args.lr_sch_pct) == len(args.lr_sch_mul)
        args.lr_sch_epochs = [int(np.floor(v * args.epochs)) for v in args.lr_sch_pct]


        args.log = not args.no_log
        if args.eval_every is None:
            args.eval_every = Config.EVAL_EVERY

        # Batch size
        args.batch_size = args.train_bs

        return args