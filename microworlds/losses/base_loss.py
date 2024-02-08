import numpy as np
import torch
import itertools
import os
from general.functions import *

class BaseLoss:
    def __init__(self, args, mask_f=torch.sigmoid):
        self.args = args
        self.mask_f = mask_f

    def sample_loss(self, sample, is_train=True):
        return NotImplementedError

    def expected_loss(self, logits, is_train=True):
        with torch.set_grad_enabled(is_train):
            return expected_value(logits, self.sample_loss, mask_function=self.mask_f)

    def expected_grad(self, logits):
        loss = self.expected_loss(logits)
        return torch.autograd.grad([loss], [logits])[0].detach()

    def get_solution(self):
        if getattr(self, 'solution', None) is None:
            with torch.no_grad():
                self.solution, _ = get_solution(self.args.num_latents, self.sample_loss)
        return self.solution
