import torch
import torch.nn as nn
from aux.parser import MainParser
from methods.base import BaseMethod
from scipy.special import logit

class CP(BaseMethod):
    class Parser(MainParser):
        def __init__(self):
            super().__init__()
            self.add_argument('--cp_initial_temp', type=float, default=1., help='Initial temperature')
            self.add_argument('--cp_final_temp', type=float, default=200., help='Final temperature')
            self.add_argument('--cp_premask_initial_value', type=float, default=0.0, help='Mask initial value')

    def __init__(self, args):
        super().__init__(args)
        self.premask = torch.zeros(self.args.d, requires_grad=True, device=self.args.device)
        cp_premask_initial_value = logit(self.args.mask_init) / self.args.cp_initial_temp
        nn.init.constant_(self.premask, cp_premask_initial_value)
        self.optim = self.optim_f([self.premask], self.args.lr)
        self.delta_temp = (self.args.cp_final_temp/self.args.cp_initial_temp) ** (1. / self.args.epochs)
        self.temp = self.args.cp_initial_temp

    def _prepare_iter(self):
        super()._prepare_iter()
        self.optim.zero_grad()
        if self.epoch_i == 0:
            self.temp = self.temp * self.delta_temp

    def _compute_and_apply_grad(self):
        self.loss = self.loss_obj.tr_loss(torch.sigmoid(self.temp*self.premask).unsqueeze(0))[0]
        self.loss.backward()
        self.optim.step()

    def val_iter(self):
        self.val_loss = self.loss_obj.val_loss((self.premask > 0).float().unsqueeze(0)).item()
        self.norm_entr = 0.0