import torch.nn.functional
from methods.monte_carlo.base import _MonteCarloLayer, MonteCarloBaseNet, MonteCarlo
from methods.base import BaseNN
import torch.nn as nn
from load_datasets.load_data import *
import torch.nn.functional as F

# NN
class _LOORFLayer(_MonteCarloLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_A = nn.Parameter(torch.zeros(self.premask.shape), requires_grad=False)
        self.acc_B = nn.Parameter(torch.zeros(self.premask.shape), requires_grad=False)

    def update_accumulator(self, this_loss):
        i = self.main_nn.accumulator_idx  # i = i + 1 is done in main model
        d_log_prob = self._cache_d_log_prob
        self.acc_A.mul_(max(i - 1, 1.) / max(1., i)).add_(d_log_prob * this_loss / max(1., i))
        self.acc_B.mul_(max(i - 1, 1.) / max(1., i)).add_(d_log_prob / max(1., i))
        self._cache_d_log_prob = None

    def compute_mc_grad(self, mean_loss):
        g = getattr(self, 'grad', None)
        this_grad = (self.acc_A - self.acc_B * mean_loss)
        if g is None:
            self.premask.grad = this_grad
        else:
            self.premask.grad.add_(this_grad)
        self.acc_A.fill_(0.0)
        self.acc_B.fill_(0.0)

    def freeze_mask(self):
        super().freeze_mask()
        self.acc_A = None
        self.acc_B = None

    def prepare_iter(self):
        pass


class _LOORFConv(_LOORFLayer, BaseNN.Conv):
    pass

class _LOORFLinear(_LOORFLayer, BaseNN.Linear):
    pass

class LOORFBaseNet(MonteCarloBaseNet):
    def _define_layer_classes(self):
        self.Conv = functools.partial(_LOORFConv, main_nn=self)
        self.Linear = functools.partial(_LOORFLinear, main_nn=self)

class LOORF(MonteCarlo):
    nn_base = LOORFBaseNet
