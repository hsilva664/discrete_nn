import torch.nn as nn
from methods.base import BaseMethod, BaseNN
from load_datasets.load_data import *
import nn_models
import argparse

# Magnitude Pruning
# NN
class MPBaseNet(BaseNN):
    def Layer(self, parent):
        class _Layer(parent):
            def __init__(self, *args, pr, **kwargs):
                super().__init__(*args, **kwargs)
                self.pr = pr
                self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)

            def prune(self, custom_pr=None):
                pr = custom_pr if custom_pr is not None else self.pr
                sorted_w = torch.sort(torch.abs(self.weight[self.mask == 1].reshape(-1)))[0]
                cutoff_idx = int(sorted_w.size()[0] * pr)
                cutoff = sorted_w[cutoff_idx]
                self.mask.data = torch.where(torch.abs(self.weight) >= cutoff, self.mask.data, torch.zeros_like(self.mask.data))

            def forward(self, x):
                _weight = self.weight
                del self.weight
                self.weight = _weight * self.mask
                o = super().forward(x)
                self.weight = _weight
                return o
        return _Layer

    def __init__(self, args):
        self.Conv = functools.partial(self.Layer(BaseNN.Conv), pr=args.conv_prune_rate)
        self.Linear = functools.partial(self.Layer(BaseNN.Linear), pr=args.fc_prune_rate)
        super().__init__(args)

    def prune(self, custom_pr=None):
        if self.args.global_prune:
            all_w = []
            global_prune_rate = custom_pr if custom_pr is not None else self.args.fc_prune_rate
            mlist = []
            for m in self.modules():
                if hasattr(m, 'mask'):
                    mlist.append(m)
                    abs_weights = torch.abs(m.weight[m.mask == 1].reshape(-1))
                    all_w.append(abs_weights)
            all_w = torch.cat(all_w)
            sorted_w = torch.sort(all_w)[0]
            cutoff_idx = int(sorted_w.size()[0] * global_prune_rate)
            cutoff = sorted_w[cutoff_idx]
            for m in mlist:
                m.mask.data = torch.where(torch.abs(m.weight) >= cutoff, m.mask.data, torch.zeros_like(m.mask.data))
        else:
            for m in self.modules():
                if hasattr(m, 'mask'):
                    m.prune(custom_pr)
# Main class
class MP(BaseMethod):
    class Parser(argparse.ArgumentParser):
        def __init__(self):
            super().__init__()
            self.add_argument('--conv_prune_rate', type=float, default=.5)
            self.add_argument('--fc_prune_rate', type=float, default=.5)
            self.add_argument('--prune_start_pct', type=float, default=.8,help='Percentage of training in which pruning should start')
            self.add_argument('--global_prune', action="store_true", default=None)

    nn_base = MPBaseNet

    def __init__(self, args):
        super().__init__(args)
        self.args.pr_epoch = int(np.floor(self.args.prune_start_pct * self.args.epochs)) - 1
        if self.args.global_prune:
            assert self.args.conv_prune_rate == self.args.fc_prune_rate

    def iter(self, *args):
        if self.state.epoch == self.args.pr_epoch and self.state.ep_iter == 0:
            self.NN.prune()
        super().iter(*args)