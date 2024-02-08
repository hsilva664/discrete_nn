import copy

import torch.nn.functional

from methods.base import BaseMethod, BaseNN
import nn_models
import torch.nn as nn
from load_datasets.load_data import *
from config import Config
from types import SimpleNamespace
import argparse
from scipy.special import logit
from scipy.special import expit as sigmoid

# Continuous Sparsification
# NN
class CPBaseNet(BaseNN):
    def Layer(self, parent):
        class _Layer(parent):
            def __init__(self, *args, model_args, **kwargs):
                super().__init__(*args, **kwargs)
                # Allow layer to keep track of model parameters (such as the current temperature)
                self.model_args = model_args
                # Init mask
                self.premask = nn.Parameter(torch.Tensor(self.weight.shape), requires_grad=True)
                if self.model_args.random_theta_0:
                    self.premask.data = logit(torch.rand_like(self.premask.data)) / self.model_args.inverse_tau_0
                else:
                    nn.init.constant_(self.premask, self.model_args.s_0)
                # Bool indicating whether weights are frozen
                self.is_frozen = False
                self.frozen_mask = None

            @property
            def mask(self):
                # Returns hard mask
                if self.is_frozen:
                    assert self.frozen_mask is not None
                    return self.frozen_mask
                else:
                    return (self.premask > 0.).float()

            @property
            def soft_mask(self):
                if self.is_frozen:
                    # Does not need soft mask (used to compute l0 reg) after masks are frozen
                    return torch.tensor([0.], device=self.model_args.device)
                else:
                    return torch.sigmoid(self.model_args.inv_tau * self.premask)

            def forward(self, x):
                _weight = self.weight
                del self.weight
                mask = self.soft_mask if self.training and not self.is_frozen else self.mask
                self.weight = _weight * mask
                o = super().forward(x)
                self.weight = _weight
                return o

            def freeze_mask(self, do_freeze_empty=False):
                if do_freeze_empty:
                    self.frozen_mask = nn.Parameter(torch.empty_like(self.weight), requires_grad=False)
                else:
                    self.frozen_mask = nn.Parameter(self.mask.detach().clone(), requires_grad=False)
                self.premask = None
                self.is_frozen = True

            def compute_and_backprop_reg_loss(self):
                if self.is_frozen: return
                soft_mask = self.soft_mask
                l0_reg_term = self.model_args.lmbda * soft_mask * self.model_args.wrem
                l0_reg_term.backward(torch.ones_like(l0_reg_term))
        return _Layer

    def __init__(self, args):
        self.Conv = functools.partial(self.Layer(BaseNN.Conv), model_args=args)
        self.Linear = functools.partial(self.Layer(BaseNN.Linear), model_args=args)
        super().__init__(args)
        self.is_frozen = False

    def freeze_masks(self, do_freeze_empty=False):
        for m in self.modules():
            if hasattr(m, 'mask'):
                m.freeze_mask(do_freeze_empty=do_freeze_empty)
        self.is_frozen = True

    def compute_and_backprop_reg_loss(self):
        for m in self.modules():
            if hasattr(m, 'mask'):
                m.compute_and_backprop_reg_loss()


# Main class
class CP(BaseMethod):
    class Parser(argparse.ArgumentParser):
        def __init__(self):
            super().__init__()
            self.add_argument('--inverse_tau_0', type=float, default=1., help='Initial temperature')
            self.add_argument('--inverse_tau_f', type=float, default=500., help='Final temperature')
            self.add_argument('--s_0', type=float, default=None, help='Premask initial value')
            self.add_argument('--theta_0', type=float, default=None, help='Mask initial value')
            self.add_argument('--random_theta_0', action="store_true")
            self.add_argument('--lmbda', type=float, default=1E-12, help='L1 regularization constant')
            self.add_argument('--ft_only_pct', type=float, default=.8, help='Percentage of training when pruning ends')
            self.add_argument('--s_lr', type=float, default=0.01, help='learning rate for premask parameters')
            self.add_argument('--s_lr_sch_mul', type=float, nargs='*', default=None, help='Multipliers for LR schedule.')
            self.add_argument('--s_optim', type=str, default="sgd", choices=["adam", "sgd", "rmsprop"], help="Mask optimizer")
            self.add_argument('--s_momentum', type=float, default=None, help='Momentum to use on mask optimizer (only works on SGD)')

        def parse_args(self, *args, **kwargs):
            args = super().parse_args(*args, **kwargs)
            if args.random_theta_0:
                assert args.theta_0 is None and args.s_0 is None
            else:
                if args.s_0 is None and args.theta_0 is None:
                    args.theta_0 = 0.5
                    args.s_0 = 0.0
                elif args.s_0 is not None and args.theta_0 is None:
                    args.theta_0 = sigmoid(args.inverse_tau_0 * args.s_0)
                elif args.s_0 is None and args.theta_0 is not None:
                    args.s_0 = logit(args.theta_0)/args.inverse_tau_0
                else:
                    assert args.s_0 == logit(args.theta_0)/args.inverse_tau_0
            return args

    nn_base = CPBaseNet

    def __init__(self, args):
        args.inv_tau = args.inverse_tau_0
        super().__init__(args)
        self.args.ft_epoch = int(np.floor(self.args.ft_only_pct * self.args.epochs))
        last_tau_epoch = self.args.ft_epoch - 1
        self.delta_inv_tau = (self.args.inverse_tau_f/self.args.inverse_tau_0) ** (1. / last_tau_epoch)

    def _filter_masked_parameters(self, return_masks):
        out = []
        for m in self.NN.modules():
            if hasattr(m, 'mask') and return_masks:
                out.append(m.premask)
            elif not return_masks:
                # PS: recurse needs to be false to avoid returning parameters more than once
                out = out + [p for n, p in m.named_parameters(recurse=False) if 'mask' not in n]
        if return_masks:
            # catch possible bugs when refactoring
            assert len(out) > 0
        return out

    def _init_optim(self):
        if self.args.train_backbone:
            self.optim = Config.get_optim(self.args, self._filter_masked_parameters(return_masks=False))
            self.all_optim = [self.optim]
        else:
            self.optim = None
            self.all_optim = []
        mask_args = SimpleNamespace(optim=self.args.s_optim,
                                    lr=self.args.s_lr,
                                    momentum=getattr(self.args, 'momentum', None),
                                    wd=None,
                                    nesterov=None
                                    )
        self.mask_optim = Config.get_optim(mask_args, self._filter_masked_parameters(return_masks=True))
        self.all_optim += [self.mask_optim]

    def _adjust_lr(self):
        if self.state.epoch in self.args.lr_sch_epochs and self.state.ep_iter == 0:
            mult_idx = self.args.lr_sch_epochs.index(self.state.epoch)
            if self.optim is not None:
                main_mul = self.args.lr_sch_mul[mult_idx]
                for param_group in self.optim.param_groups:
                    param_group['lr'] = param_group['lr'] * main_mul
            # Check if mask lr decay was requested
            if self.mask_optim is not None and self.args.s_lr_sch_mul is not None:
                mask_mul = self.args.s_lr_sch_mul[mult_idx]
                for param_group in self.mask_optim.param_groups:
                    param_group['lr'] = param_group['lr'] * mask_mul

    def _prepare_iter(self, is_train):
        super()._prepare_iter(is_train)
        if is_train:
            if self.state.ep_iter == 0 and self.state.epoch < self.args.ft_epoch:
                self.args.inv_tau = self.args.inverse_tau_0 * (self.delta_inv_tau ** self.state.epoch)

    def _compute_grad(self, *args):
        super()._compute_grad(*args)
        self.NN.compute_and_backprop_reg_loss()

    def iter(self, *args):
        if self.state.epoch == self.args.ft_epoch and self.state.ep_iter == 0:
            self.freeze_masks()
        super().iter(*args)

    def freeze_masks(self, do_freeze_empty=False):
        self.NN.freeze_masks(do_freeze_empty)
        self.all_optim.remove(self.mask_optim)
        self.mask_optim=None

    def set_state(self, source_dict):
        self.NN.is_frozen = source_dict["is_frozen"]
        if self.NN.is_frozen:
            self.freeze_masks(do_freeze_empty=True)
        super().set_state(source_dict)

    def get_state(self, target_dict):
        target_dict["is_frozen"] = self.NN.is_frozen
        super().get_state(target_dict)
