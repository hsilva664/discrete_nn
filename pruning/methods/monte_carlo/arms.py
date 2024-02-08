import torch.nn.functional
from methods.monte_carlo.base import _MonteCarloLayer, MonteCarloBaseNet, MonteCarlo
from methods.base import BaseNN
import torch.nn as nn
from load_datasets.load_data import *
import torch.nn.functional as F

# NN
class _ARMSLayer(_MonteCarloLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acc_A = nn.Parameter(torch.zeros(self.premask.shape), requires_grad=False)
        self.acc_B = nn.Parameter(torch.zeros(self.premask.shape), requires_grad=False)
        self.gamma = torch.distributions.gamma.Gamma(concentration=torch.tensor(self.model_args.n,
                                                                                dtype=torch.get_default_dtype(),
                                                                                device=self.main_nn.args.device),
                                                      rate=torch.tensor(1,
                                                                        dtype=torch.get_default_dtype(),
                                                                         device=self.main_nn.args.device)
                                                     )
        self.rho = None
        self.sum_Ei = None
        self.all_R = None

    def prepare_iter(self):
        if self.main_nn.is_frozen or self.main_nn.is_pretraining:
            return
        self.rho = torch.where((self._soft_mask == 0.0) | (self._soft_mask == 1.0), torch.tensor(0., device=self._soft_mask.device),
                               torch.where(self._soft_mask > 0.5, self._rho_f(self._soft_mask), self._rho_fp(self._soft_mask)))
        # Allows broadcasting to differently-shaped parameters (i.e. for Escort)
        self.rho = self.rho.reshape(
            list(self.rho.shape) + [1 for _ in range(len(self.premask.shape) - len(self.rho.shape))])
        self.sum_Ei = self.gamma.sample(self._soft_mask.shape)
        self.all_R = self.sum_Ei.detach().clone()

    @property
    def indexed_sampled_mask(self):
        with torch.no_grad():
            ip1 = self.main_nn.accumulator_idx + 1
            if ip1 < self.model_args.n:
                Ei = -self.all_R * torch.rand_like(self._soft_mask) \
                     ** (1 / (self.model_args.n - ip1)) + self.all_R
            elif ip1 == self.model_args.n:
                Ei = self.all_R.detach().clone()
            else:
                raise IndexError
            self.all_R -= Ei
            cur_d = Ei / self.sum_Ei
            cur_u_tilde = 1 - (1 - cur_d) ** (self.model_args.n - 1)
            cur_eff_u = torch.where(self._soft_mask > 0.5, cur_u_tilde, 1 - cur_u_tilde)
            return cur_eff_u.lt(self._soft_mask).type_as(cur_eff_u)
        
    def update_accumulator(self, this_loss):
        i = self.main_nn.accumulator_idx  # i = i + 1 is done in main model
        d_log_prob = self._cache_d_log_prob
        self.acc_A.mul_(max(i - 1, 1.) / max(1., i)).add_(d_log_prob * this_loss / max(1., i))
        self.acc_B.mul_(max(i - 1, 1.) / max(1., i)).add_(d_log_prob / max(1., i))
        self._cache_d_log_prob = None

    def compute_mc_grad(self, mean_loss):
        g = getattr(self, 'grad', None)
        this_grad = (self.acc_A - self.acc_B * mean_loss)/(1 - self.rho)
        if g is None:
            self.premask.grad = this_grad
        else:
            self.premask.grad.add_(this_grad)
        self.acc_A.fill_(0.0)
        self.acc_B.fill_(0.0)
        self.rho = None
        self.sum_Ei = None
        self.all_R = None

    def freeze_mask(self):
        super().freeze_mask()
        self.acc_A = None
        self.acc_B = None
        self.rho = None
        self.sum_Ei = None
        self.all_R = None
        
    def _rho_f(self, probs):
        return (torch.maximum(torch.zeros_like(probs),
                              2 * (1 - probs) ** (1 / (self.model_args.n - 1)) - 1) ** (
                        self.model_args.n - 1) - (1 - probs) ** 2) / \
               (probs * (1 - probs))


    def _rho_fp(self, probs):
        return (torch.maximum(torch.zeros_like(probs),
                              2 * probs ** (1 / (self.model_args.n - 1)) - 1) ** (
                        self.model_args.n - 1) - probs ** 2) / \
               (probs * (1 - probs))


class _ARMSConv(_ARMSLayer, BaseNN.Conv):
    pass

class _ARMSLinear(_ARMSLayer, BaseNN.Linear):
    pass

class ARMSBaseNet(MonteCarloBaseNet):
    def _define_layer_classes(self):
        self.Conv = functools.partial(_ARMSConv, main_nn=self)
        self.Linear = functools.partial(_ARMSLinear, main_nn=self)

class ARMS(MonteCarlo):
    nn_base = ARMSBaseNet
