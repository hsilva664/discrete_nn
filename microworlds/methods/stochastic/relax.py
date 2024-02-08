import torch
import numpy as np
from general.functions import *
from methods.stochastic.base import *

class Relax(Stochastic):
    # 1 forward pass for b, 1 fp and 1 bp for both z and zt
    fp_bp_n = 5

    def __init__(self, args, loss_obj):
        super().__init__(args, loss_obj)

        self.log_temp = torch.from_numpy(
            np.array([.5] * args.num_latents)).type_as(self.logits)
            
        self.log_temp.requires_grad_(True)

        self.q_func = QFunc(args.num_latents, hidden_size=args.hidden_size, nlayers=args.n_layers)

        self.tunable = [self.log_temp] + list(self.q_func.parameters())

        self.tune_optim = torch.optim.Adam([{"params": [self.log_temp]},
                                            {"params": self.q_func.parameters(), "lr": 1.0, "weight_decay": 0.001}], lr=args.lr)

        self.all_optim += [self.tune_optim]

    def _prepare_iter(self):
        super()._prepare_iter()
        self.v = v_from_u(self.u, self.cur_logit_batch)

    def _compute_grad(self):
        zt = self.cur_logit_batch + safe_log_prob(self.v) - safe_log_prob_1p(-self.v)
        temp = torch.exp(self.log_temp).unsqueeze(0)
        sig_z = torch.sigmoid(self.z / temp)
        sig_zt = torch.sigmoid(zt / temp)
        q_sig_z = self.q_func(sig_z)[:, 0]
        q_sig_zt = self.q_func(sig_zt)[:, 0]
        log_prob = torch.distributions.Bernoulli(logits=self.cur_logit_batch).log_prob(self.b)
        d_log_prob = torch.autograd.grad(
            [log_prob], [self.cur_logit_batch], grad_outputs=torch.ones_like(log_prob))[0]
        d_q_sig_z = torch.autograd.grad(
            [q_sig_z], [self.cur_logit_batch], grad_outputs=torch.ones_like(q_sig_z),
            create_graph=True, retain_graph=True)[0]
        d_q_sig_zt = torch.autograd.grad(
            [q_sig_zt], [self.cur_logit_batch], grad_outputs=torch.ones_like(q_sig_zt),
            create_graph=True, retain_graph=True)[0]
        diff = self.loss.unsqueeze(1) - q_sig_zt.unsqueeze(1)
        d_logits = diff * d_log_prob + d_q_sig_z - d_q_sig_zt
        var_loss = (d_logits.mean(0) ** 2).mean()           
        d_tunable = torch.autograd.grad(
            [var_loss], self.tunable, grad_outputs=torch.ones_like(var_loss))

        return {'logits': ([self.logits], [d_logits.detach()]),
                'log_temp': ([self.log_temp], [d_tunable[0].detach()]),
                'q_func': (list(self.q_func.parameters()), list(d_tunable[1:]) )
                }