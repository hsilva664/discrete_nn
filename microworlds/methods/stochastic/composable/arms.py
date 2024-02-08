import torch
import numpy as np
from general.functions import *
from methods.stochastic.composable.base import Composable

class ARMS(Composable):

    def is_norm_implemented(self):
        return True

    def _rho_f(self, p):
        return (torch.maximum(torch.zeros_like(p) , 2*(1-p)**(1/(self.args.batch_size-1)) - 1)**(self.args.batch_size-1) - (1-p)**2)/ \
                (p * (1-p))

    def _rho_fp(self, p):
        return (torch.maximum(torch.zeros_like(p), 2*p**(1/(self.args.batch_size-1)) - 1)**(self.args.batch_size-1) - p**2)/ \
                (p * (1-p))

    def _prepare_iter(self, input_z_batch=None):
        assert self.args.batch_size > 1
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()

        with torch.no_grad():
            # Prepare input batch
            ## (full batch required for batch_size > 1, otherwise autograd will sum the gradients before the correct time and it will not be possible to calculate the mean and std of the gradient estimator)
            self.cur_logit_batch = self.logits.detach().unsqueeze(0).expand([self.args.batch_size]+list(self.logits.shape))
            self.probs = self.logits_to_probs.f(self.cur_logit_batch)
            self.rho = torch.where((self.probs == 0) | (self.probs == 1), torch.tensor(0.),
                                   torch.where(self.probs > 0.5, self._rho_f(self.probs), self._rho_fp(self.probs)))
            self.rho = self.rho.reshape(list(self.rho.shape) + [1 for _ in range(len(self.cur_logit_batch.shape) - len(self.rho.shape))])

            if input_z_batch is None:
                # Uniform variable and bernoulli sample
                raw_u = torch.rand(self.args.batch_size, self.args.num_latents)
                d = torch.log(raw_u) / torch.sum( torch.log(raw_u), 0)
                u = 1 - (1-d)**(self.args.batch_size - 1)
                eff_u = torch.where( self.probs > 0.5, u, 1 - u )
                self.z_batch = eff_u.le( self.probs ).type_as(eff_u)
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
            mean_loss = loss.mean(0, keepdim=True)
            d_log_prob = self.logits_to_probs.d_log_prob(self.z_batch, self.prob_z_batch, self.cur_logit_batch, self.probs)
            inner_term = d_log_prob/ (1 - self.rho)
            d_logits = (1/(self.args.batch_size - 1.))*torch.sum(((loss - mean_loss) * inner_term), 0, keepdim=True)

        return {'logits': ([self.logits], [d_logits])}