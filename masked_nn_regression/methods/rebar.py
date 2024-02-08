import torch
import numpy as np
from scipy.special import logit
from aux.parser import MainParser
from methods.base import BaseMethod
from torch.utils.data import DataLoader

def safe_clip(x, eps=1e-8):
    return torch.clip(x, eps, 1.0)

def safe_log_prob(x, eps=1e-8):
    return torch.log(safe_clip(x, eps))

def safe_log_prob_1p(x, eps=1e-8):
    return torch.log1p(torch.clip(x, -(1 - eps), 0.0))

def v_from_u(u, logits):
    u_prime = torch.sigmoid(-logits)
    with torch.no_grad():
        v_1 = (u - u_prime) / safe_clip(1 - u_prime)
        v_1 = torch.clip(v_1, 0, 1)
        v_0 = u / safe_clip(u_prime)
        v_0 = torch.clip(v_0, 0, 1)

    v_1 = v_1 * (1 - u_prime) + u_prime
    v_0 = v_0 * u_prime

    v = torch.where(u > u_prime, v_1, v_0)
    v = v + (-v + u).detach()
    return v

class Rebar(BaseMethod):
    class Parser(MainParser):
        def __init__(self):
            super().__init__()
            self.add_argument('--n', type=int, default=10,
                               help='Here each n corresponds to 3 forward passes and 2 backward passes')

    def __init__(self, args):
        super().__init__(args)
        self.premask = torch.zeros(self.args.d, requires_grad=True, device=self.args.device)
        torch.nn.init.constant_(self.premask, logit(self.args.mask_init))
        self.optim = self.optim_f([self.premask], self.args.lr)

        self.eta = torch.ones(args.d, requires_grad=True, device=self.args.device)
        self.log_temp = torch.from_numpy(
            np.array([.5] * args.d)).type_as(self.premask).to(self.args.device)
        self.log_temp.requires_grad_(True)
        self.tune_optim = torch.optim.Adam([self.eta, self.log_temp], lr=self.args.lr)

    def _prepare_iter(self):
        super()._prepare_iter()
        self.optim.zero_grad()
        self.tune_optim.zero_grad()
        # Compute correlation
        self.sig_logit = torch.sigmoid(self.premask)
        self.cur_logit_batch = self.premask.detach().unsqueeze(0).repeat(self.args.n, 1)
        self.cur_logit_batch.requires_grad = True
        # PS: here, "z" from composable methods is called "b" instead, for consistency with the original REBAR paper
        self.u = torch.rand(self.args.n, *self.premask.shape).to(self.args.device)
        self.z = self.cur_logit_batch + safe_log_prob(self.u) - safe_log_prob_1p(-self.u)
        self.b = self.z.gt(0.).type_as(self.z)
        self.v = v_from_u(self.u, self.cur_logit_batch)

    def _compute_d_log_prob(self, b):
        with torch.no_grad():
            return b - self.sig_logit

    def _compute_and_apply_grad(self):
        zt = self.cur_logit_batch + safe_log_prob(self.v) - safe_log_prob_1p(-self.v)
        temp = torch.exp(self.log_temp).unsqueeze(0)
        sig_z = torch.sigmoid(self.z / temp)
        sig_zt = torch.sigmoid(zt / temp)
        loss = self.loss_obj.tr_loss(self.b)
        loss_sig_z = self.loss_obj.tr_loss(sig_z)
        loss_sig_zt = self.loss_obj.tr_loss(sig_zt)
        d_log_prob = self._compute_d_log_prob(self.b)
        d_f_z = torch.autograd.grad(
            [loss_sig_z], [self.cur_logit_batch], grad_outputs=torch.ones_like(loss_sig_z),
            create_graph=True, retain_graph=True)[0]
        d_f_z_tilde = torch.autograd.grad(
            [loss_sig_zt], [self.cur_logit_batch], grad_outputs=torch.ones_like(loss_sig_zt),
            create_graph=True, retain_graph=True)[0]
        diff = loss.unsqueeze(1) - self.eta * loss_sig_zt.unsqueeze(1)
        d_logits = diff * d_log_prob + self.eta * (d_f_z - d_f_z_tilde)
        var_loss = (d_logits ** 2).mean()
        d_eta, d_log_temp = torch.autograd.grad(
            [var_loss], [self.eta, self.log_temp], grad_outputs=torch.ones_like(var_loss))

        d_eta = d_eta.detach()
        d_log_temp = d_log_temp.detach()
        d_logits = d_logits.mean(0).detach()

        self.eta.backward(d_eta)
        self.log_temp.backward(d_log_temp)
        self.premask.backward(d_logits)
        self.optim.step()
        self.tune_optim.step()

    def get_val_mask(self, probs):
        with torch.no_grad():
            # PS: for REBAR, there is no temp on main forward pass, only on auxiliary control variates
            THIS_BATCH_SIZE = 5 * self.loss_obj.batch_size
            THIS_ATTEMPTED_MASKS = 5
            dataset = self.loss_obj.dataset
            loader = DataLoader(dataset=dataset, batch_size=THIS_BATCH_SIZE, shuffle=True)
            x, y = next(iter(loader))
            curr_min_b = None
            curr_min_loss = np.Inf
            for _ in range(THIS_ATTEMPTED_MASKS):
                u = torch.rand(*probs.shape).to(self.args.device)
                b = u.lt(probs).type_as(probs)
                this_loss = self.loss_obj.tr_loss(b.unsqueeze(0), x=x, y=y, force_eval_mode=True).item()
                if this_loss < curr_min_loss:
                    curr_min_loss = this_loss
                    curr_min_b = b
            return curr_min_b

    def val_iter(self):
        probs = torch.sigmoid(self.premask)
        b = self.get_val_mask(probs)
        device = probs.device
        dtype = probs.dtype
        pre_entr = -probs * torch.log(probs) -(1 - probs) * torch.log(1 - probs)
        self.norm_entr = (torch.where( (probs != 0.0) & (probs != 1.0), pre_entr, torch.tensor(0.0, device=device, dtype=dtype)).sum() /
                          (self.args.d * torch.log(torch.tensor(2.0, device=device, dtype=dtype)))).item()
        self.val_loss = self.loss_obj.val_loss(b.unsqueeze(0)).item()
