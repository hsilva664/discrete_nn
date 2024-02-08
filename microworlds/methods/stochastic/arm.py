import torch
import numpy as np
from general.functions import *
from methods.stochastic.base import *

class ARM(Stochastic):
    fp_bp_n = 2

    def __init__(self, args, loss_obj):
        super().__init__(args, loss_obj)

    def _compute_grad(self):
        with torch.no_grad():
            u_gt_neg_sig = self.u.gt(torch.sigmoid(-self.logits)).type_as(self.logits)
            u_lt_sig = self.u.le(torch.sigmoid(self.logits)).type_as(self.logits)
            loss_neg = self.loss_obj.sample_loss(u_gt_neg_sig, is_train=False).unsqueeze(1)
            loss_pos = self.loss_obj.sample_loss(u_lt_sig, is_train=False).unsqueeze(1)
            d_logits = ((loss_neg - loss_pos)*(self.u - 0.5))
            return {'logits': ([self.logits], [d_logits])}