import torch
import torch.nn as nn
from aux.parser import MainParser
from methods.base import BaseMethod
from scipy.special import logit
from torch.utils.data import DataLoader
import numpy as np

class ST(BaseMethod):
    class Parser(MainParser):
        def __init__(self):
            super().__init__()
            self.add_argument('--n', type=int, default=10,
                               help='Here each n corresponds to 1 forward pass and 1 backward pass')

    def __init__(self, args):
        super().__init__(args)
        self.premask = torch.zeros(self.args.d, requires_grad=True, device=self.args.device)
        premask_initial_value = logit(self.args.mask_init)
        nn.init.constant_(self.premask, premask_initial_value)
        self.optim = self.optim_f([self.premask], self.args.lr)

    def _prepare_iter(self):
        super()._prepare_iter()
        self.optim.zero_grad()

    def _compute_and_apply_grad(self):
        self.probs = torch.sigmoid(self.premask)
        self.loss = 0.0
        x = None
        y = None
        for i in range(self.args.n):
            raw_u = torch.rand_like(self.probs)
            z = raw_u.lt(self.probs).type_as(self.probs).detach()
            z.requires_grad = True
            if i == 0:
                this_loss, x, y = self.loss_obj.tr_loss(z.unsqueeze(0), return_input=True)
            else:
                this_loss = self.loss_obj.tr_loss(z.unsqueeze(0), x=x, y=y)
            if self.premask.grad is None:
                self.premask.grad = torch.autograd.grad([this_loss], [z], grad_outputs=torch.ones_like(this_loss))[0]/self.args.n
            else:
                self.premask.grad.add_(torch.autograd.grad([this_loss], [z], grad_outputs=torch.ones_like(this_loss))[0]/self.args.n)
            self.loss = (i / (i + 1)) * self.loss + (1. / (i + 1)) * this_loss.item()

        self.premask.grad.mul_(self.probs*(1-self.probs))
        self.optim.step()

    def get_val_mask(self, probs):
        with torch.no_grad():
            THIS_BATCH_SIZE = 5 * self.loss_obj.batch_size
            THIS_ATTEMPTED_MASKS = 5
            dataset = self.loss_obj.dataset
            loader = DataLoader(dataset=dataset, batch_size=THIS_BATCH_SIZE, shuffle=True)
            x, y = next(iter(loader))
            curr_min_z = None
            curr_min_loss = np.Inf
            for _ in range(THIS_ATTEMPTED_MASKS):
                u = torch.rand(*probs.shape).to(self.args.device)
                z = u.lt(probs).type_as(probs)
                this_loss = self.loss_obj.tr_loss(z.unsqueeze(0), x=x, y=y, force_eval_mode=True).item()
                if this_loss < curr_min_loss:
                    curr_min_loss = this_loss
                    curr_min_z = z
            return curr_min_z

    def val_iter(self):
        probs = torch.sigmoid(self.premask)
        z = self.get_val_mask(probs)
        device = probs.device
        dtype = probs.dtype
        pre_entr = -probs * torch.log(probs) -(1 - probs) * torch.log(1 - probs)
        self.norm_entr = (torch.where( (probs != 0.0) & (probs != 1.0), pre_entr, torch.tensor(0.0, device=device, dtype=dtype)).sum() /
                          (self.args.d * torch.log(torch.tensor(2.0, device=device, dtype=dtype)))).item()
        self.val_loss = self.loss_obj.val_loss(z.unsqueeze(0)).item()