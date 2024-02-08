import torch
import numpy as np
from general.functions import *
from methods.stochastic.composable.base import Composable

class ARMSIter(Composable):
    def __init__(self, args, loss_obj):
        super().__init__(args, loss_obj)
        self.acc_A = torch.zeros_like(self.logits)
        self.acc_B = torch.zeros_like(self.logits)
        self.gamma = torch.distributions.gamma.Gamma(self.n, 1)
        self.uniform = torch.distributions.uniform.Uniform(0.0, 1.0)

    def _rho_f(self, p):
        return (torch.maximum(torch.zeros_like(p) , 2*(1-p)**(1/(self.n-1)) - 1)**(self.n-1) - (1-p)**2)/ \
                (p * (1-p))

    def _rho_fp(self, p):
        return (torch.maximum(torch.zeros_like(p), 2*p**(1/(self.n-1)) - 1)**(self.n-1) - p**2)/ \
                (p * (1-p))

    def _prepare_iter(self):
        assert self.n > 1
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()

        with torch.no_grad():
            self.probs = self.logits_to_probs.f(self.logits)
            self.sum_Ei = self.gamma.sample((self.d,))
            self.all_R = self.sum_Ei.detach().clone()

        # Calculate true grad
        if self.args.use_true_grad:
            self.true_grad = self.loss_obj.expected_grad(self.logits)

    def _compute_grad(self):
        with torch.no_grad():
            gi = 0
            self.loss = 0.0
            rho = torch.where(self.probs > 0.5, self._rho_f(self.probs), self._rho_fp(self.probs))
            rho = rho.reshape(
                list(rho.shape) + [1 for _ in range(len(self.logits.shape) - len(rho.shape))])
            for i in range(1, self.n+1):
                if i < self.n:
                    Ei = -self.all_R * self.uniform.sample((self.d,)) \
                         ** (1 / (self.n - i)) + self.all_R
                else:
                    Ei = self.all_R.detach().clone()
                self.all_R -= Ei
                cur_d = Ei / self.sum_Ei
                cur_u_tilde = 1 - (1-cur_d)**(self.n - 1)
                cur_eff_u = torch.where(self.probs > 0.5, cur_u_tilde, 1 - cur_u_tilde)
                cur_z = cur_eff_u.le(self.probs).type_as(cur_eff_u)
                cur_prob_z = torch.where(cur_z == 1, self.probs, 1 - self.probs)

                cur_loss = self.loss_obj.sample_loss(cur_z.unsqueeze(0), is_train=False)
                d_log_prob = self.logits_to_probs.d_log_prob(cur_z, cur_prob_z, self.logits, self.probs)
                inner_term = d_log_prob / (1 - rho)

                self.acc_A.mul_(max(gi - 1, 1.)/max(1., gi)).add_(inner_term * cur_loss/max(1., gi))
                self.acc_B.mul_(max(gi - 1, 1.)/max(1., gi)).add_(inner_term/max(1., gi))
                self.loss = (gi / (gi + 1)) * self.loss + (1. / (gi + 1)) * cur_loss

                gi += 1

            d_logits = self.acc_A - self.acc_B * self.loss
            self.acc_A.fill_(0.0)
            self.acc_B.fill_(0.0)

        return {'logits': ([self.logits], [d_logits.unsqueeze(0)])}