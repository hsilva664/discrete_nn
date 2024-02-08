import torch
import numpy as np
from general.functions import *
from methods.stochastic.composable.base import Composable

class BetaStar(Composable):
    def __init__(self, args, loss_obj):
        super().__init__(args, loss_obj)

    def is_norm_implemented(self):
        return True

    def _prepare_iter(self, input_z_batch=None):
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()

        # Prepare input batch
        ## (full batch required for batch_size > 1, otherwise autograd will sum the gradients before the correct time,
        # making it impossible to calculate the mean and std of the gradient estimator)
        self.cur_logit_batch = self.logits.detach().unsqueeze(0).expand([self.n] + list(self.logits.shape))
        self.probs = self.logits_to_probs.f(self.cur_logit_batch)

        if input_z_batch is None:
            # Uniform variable and bernoulli sample
            u = self._sample_uniform(self.n, self.args.num_latents)
            self.z_batch = u.lt(self.probs).type_as(self.probs)
        else:
            self.z_batch = input_z_batch

        self.prob_z_batch = torch.where(self.z_batch == 1, self.probs, 1 - self.probs)

        self.loss = self.loss_obj.sample_loss(self.z_batch, is_train=False)
        # Calculate true grad
        if self.args.use_true_grad:
            self.true_grad = self.loss_obj.expected_grad(self.logits)

    def _compute_grad(self):
        # PS: variance is going to be wrong in the log, since the estimator already merges the batch
        assert self.n > 1
        d_log_prob = self.logits_to_probs.d_log_prob(self.z_batch, self.prob_z_batch, self.cur_logit_batch, self.probs)

        if self.args.normalize_loss:
            loss = self.normalize_loss()
        else:
            loss = self.loss

        with torch.no_grad():
            beta = torch.zeros_like(self.prob_z_batch)
            def beta_f(z, i):
                ## IS version
                # probs = self.logits_to_probs.f(self.logits)
                # prob_z = torch.where(z == 1, probs, 1 - probs)
                # prob_zi = prob_z[:, i]
                # loss = self.loss_obj.sample_loss(z, is_train=False)
                # out = ((1 - prob_zi) / prob_zi) * loss
                z_cp = z.detach().clone()
                z_cp.data[:, i].mul_(-1).add_(1)
                out = self.loss_obj.sample_loss(z_cp, is_train=False)
                return out
            import functools
            for i in range(self.d):
                beta_i = functools.partial(beta_f, i=i)
                b_star = expected_value(self.logits, beta_i, mask_function=self.logits_to_probs.f)
                beta[:, i] = b_star

            inner_term = (loss.unsqueeze(1) - beta)
            inner_term = inner_term.reshape(
                list(inner_term.shape) + [1 for _ in range(len(self.cur_logit_batch.shape) - len(inner_term.shape))])

            d_logits = (1/self.n)*torch.sum(d_log_prob * inner_term, 0, keepdim=True)

        return {'logits': ([self.logits], [d_logits])}