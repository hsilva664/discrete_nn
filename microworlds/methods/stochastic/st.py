import torch
import numpy as np
from general.functions import *
from methods.stochastic.base import *

class ST(Stochastic):
    # 1 forward pass, 1 backward pass
    fp_bp_n = 2

    def __init__(self, args, loss_obj):
        super().__init__(args, loss_obj)

    def _compute_grad(self):
        sig = torch.sigmoid(self.cur_logit_batch)
        b = self.b.detach()
        b.requires_grad = True
        f_b = self.loss_obj.sample_loss(b)
        d_f_b = torch.autograd.grad( [f_b], [b], grad_outputs=torch.ones_like(f_b))[0]
        d_logits = sig * (1-sig) * d_f_b
        return {'logits': ([self.logits], [d_logits.detach()])}