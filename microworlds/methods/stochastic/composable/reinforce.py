import torch
import numpy as np
from general.functions import *
from methods.stochastic.composable.base import Composable

class Reinforce(Composable):
    def is_norm_implemented(self):
        return True

    def _prepare_iter(self, input_z_batch=None):
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()

        # Prepare input batch
        ## (full batch required for batch_size > 1, otherwise autograd will sum the gradients before the correct time,
        # making it impossible to calculate the mean and std of the gradient estimator)
        self.cur_logit_batch = self.logits.detach().unsqueeze(0).expand([self.args.batch_size] + list(self.logits.shape))
        self.probs = self.logits_to_probs.f(self.cur_logit_batch)

        if input_z_batch is None:
            # Uniform variable and bernoulli sample
            u = self._sample_uniform(self.args.batch_size, self.args.num_latents)
            self.z_batch = u.lt(self.probs).type_as(self.probs)
        else:
            self.z_batch = input_z_batch

        self.prob_z_batch = torch.where(self.z_batch == 1, self.probs, 1 - self.probs)

        self.loss = self.loss_obj.sample_loss(self.z_batch, is_train=False)
        # Calculate true grad
        if self.args.use_true_grad:
            self.true_grad = self.loss_obj.expected_grad(self.logits)

    def _compute_grad(self):
        if self.args.normalize_loss:
            loss = self.normalize_loss()
        else:
            loss = self.loss

        loss = loss.reshape([self.n] + [1 for _ in range(len(self.logits.shape))])
        with torch.no_grad():
            d_log_prob = self.logits_to_probs.d_log_prob(self.z_batch, self.prob_z_batch, self.cur_logit_batch, self.probs)
            d_logits = loss * d_log_prob

        return {'logits': ([self.logits], [d_logits])}