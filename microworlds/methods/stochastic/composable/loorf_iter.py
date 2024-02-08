import torch
import numpy as np
from general.functions import *
from methods.stochastic.composable.base import Composable


class LOORFIter(Composable):
    def __init__(self, args, loss_obj):
        super().__init__(args, loss_obj)
        # Accumulators for iterative implementation
        self.acc_A = torch.zeros_like(self.logits)
        self.acc_B = torch.zeros_like(self.logits)

    def _prepare_iter(self):
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()

        with torch.no_grad():
            self.probs = self.logits_to_probs.f(self.logits)

        # Calculate true grad
        if self.args.use_true_grad:
            self.true_grad = self.loss_obj.expected_grad(self.logits)


    def _compute_grad(self):
        # PS: variance is going to be wrong in the log, since the estimator already merges the batch
        assert self.args.batch_size > 1

        with torch.no_grad():
            self.loss = 0.0

            for i in range(self.args.batch_size):
                this_u = torch.rand_like(self.probs)
                cur_z = this_u.le(self.probs).type_as(self.probs)
                cur_prob_z = torch.where(cur_z == 1, self.probs, 1 - self.probs)
                d_log_prob = self.logits_to_probs.d_log_prob(cur_z, cur_prob_z, self.logits, self.probs)
                this_loss = self.loss_obj.sample_loss(cur_z.unsqueeze(0), is_train=False)

                self.acc_A.mul_(max(i - 1, 1.)/max(1., i)).add_(d_log_prob * this_loss/max(1., i))
                self.acc_B.mul_(max(i - 1, 1.)/max(1., i)).add_(d_log_prob/max(1., i))
                self.loss = (i/(i+1)) * self.loss + (1./(i + 1)) * this_loss

            d_logits = self.acc_A - self.acc_B * self.loss
            self.acc_A.fill_(0.0)
            self.acc_B.fill_(0.0)
        return {'logits': ([self.logits], [d_logits.unsqueeze(0)])}
