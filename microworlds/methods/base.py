import torch
import numpy as np
from general.functions import *
import copy

class Base:
    # Number of forward and backward passes
    fp_bp_n = 1

    def __init__(self, args, loss_obj):
        self.args = copy.deepcopy(args)
        assert self.args.logits_to_probs_str is None
        self.loss_obj = loss_obj
        self.logits = torch.zeros(args.num_latents, requires_grad=True)
        self.logit_optim = torch.optim.SGD([self.logits], lr=args.lr)
        self.all_optim = [self.logit_optim]
        self._correct_batch_size()
        self.n = self.args.batch_size
        self.d = self.args.num_latents

    def normalize_loss(self):
        # Not all methods are compatible with this
        if self.loss.max() != self.loss.min():
            norm_loss = (self.loss - self.loss.mean())/self.loss.std()
        else:
            norm_loss = torch.zeros_like(self.loss)
        return norm_loss

    def is_norm_implemented(self):
        raise NotImplementedError("Loss normalization is not implemented for {}".format(self.args.estimator))

    def _correct_batch_size(self):
        if self.args.make_computation_equivalent:
            # fb_bp_n = number of forward and backward passes for each "1" evaluation
            assert self.args.batch_size % self.fp_bp_n == 0
            self.args.batch_size = self.args.batch_size // self.fp_bp_n

    def est_str(self):
        return self.args.estimator

    def iter(self, iteration):
        self._prepare_iter()
        grad_dict = self._compute_grad()
        self._apply_grad(grad_dict)
        if iteration % self.args.log_steps == 0:
            self._compute_stats_and_log(iteration, grad_dict)

    def _prepare_iter(self):
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()

        # Prepare input batch 
        ## (full batch required for batch_size > 1, otherwise autograd will sum the gradients before the correct time,
        # making it impossible to calculate the mean and std of the gradient estimator)
        self.cur_logit_batch = self.logits.detach().unsqueeze(0).repeat(self.args.batch_size,1)
        self.cur_logit_batch.requires_grad = True

        # Uniform variable and bernoulli sample
        self.u = self._sample_uniform(self.args.batch_size, self.args.num_latents)
        self.z = self.cur_logit_batch + safe_log_prob(self.u) - safe_log_prob_1p(-self.u)
        self.b = self.z.gt(0.).type_as(self.z)

        self.loss = self.loss_obj.sample_loss(self.b, is_train=False)
        # Calculate true grad
        if self.args.use_true_grad:
            self.true_grad = self.loss_obj.expected_grad(self.logits)

    def _sample_uniform(self, *args):
        return torch.rand(*args)

    def _compute_grad(self):
        raise NotImplementedError

    def _apply_grad(self, grad_map):
        for var_type, (param_list, grad_list) in grad_map.items():
            for i in range(len(param_list)):
                if var_type == 'logits' and self.args.use_true_grad:
                    param_list[i].backward(self.true_grad)
                else:
                    if var_type == 'logits':
                        param_list[i].backward(grad_list[i].mean(0))
                    else:
                        param_list[i].backward(grad_list[i])

        for optim in self.all_optim:
            optim.step()

    def _compute_expected_loss_for_logger(self):
        return self.loss_obj.expected_loss(self.logits, is_train=False).item()

    def _compute_stats_and_log(self, iteration, grad_dict):
        assert len(grad_dict['logits'][0]) == 1
        thetas = self.loss_obj.mask_f(self.logits.detach())
        logger_thetas = thetas.numpy()
        expected_loss = self._compute_expected_loss_for_logger()
        xnor_d = xnor_distance(thetas, self.loss_obj.get_solution()).item()
        self.logger.log([iteration, expected_loss, xnor_d, self.logits.detach().numpy(), logger_thetas])
        