from methods.stochastic.base import *

class Rebar(Stochastic):
    # 1 forward pass for b, 1 fp and 1 bp for both z and zt
    fp_bp_n = 5

    def __init__(self, args, loss_obj):
        super().__init__(args, loss_obj)
        self.eta = torch.ones(args.num_latents, requires_grad=True)

        self.log_temp = torch.from_numpy(
            np.array([.5] * args.num_latents)).type_as(self.logits)
            
        self.log_temp.requires_grad_(True)

        self.tunable = [self.eta, self.log_temp]

        self.tune_optim = torch.optim.Adam(self.tunable, lr=args.lr)

        self.all_optim += [self.tune_optim]

    def _prepare_iter(self):
        super()._prepare_iter()
        self.v = v_from_u(self.u, self.cur_logit_batch)

    def _compute_grad(self):
        zt = self.cur_logit_batch + safe_log_prob(self.v) - safe_log_prob_1p(-self.v)
        temp = torch.exp(self.log_temp).unsqueeze(0)
        sig_z = torch.sigmoid(self.z / temp)
        sig_zt = torch.sigmoid(zt / temp)
        loss_sig_z = self.loss_obj.sample_loss(sig_z)
        loss_sig_zt = self.loss_obj.sample_loss(sig_zt)
        log_prob = torch.distributions.Bernoulli(logits=self.cur_logit_batch).log_prob(self.b)
        d_log_prob = torch.autograd.grad(
            [log_prob], [self.cur_logit_batch], grad_outputs=torch.ones_like(log_prob))[0]
        d_f_z = torch.autograd.grad(
            [loss_sig_z], [self.cur_logit_batch], grad_outputs=torch.ones_like(loss_sig_z),
            create_graph=True, retain_graph=True)[0]
        d_f_z_tilde = torch.autograd.grad(
            [loss_sig_zt], [self.cur_logit_batch], grad_outputs=torch.ones_like(loss_sig_zt),
            create_graph=True, retain_graph=True)[0]
        diff = self.loss.unsqueeze(1) - self.eta * loss_sig_zt.unsqueeze(1)
        d_logits = diff * d_log_prob + self.eta * (d_f_z - d_f_z_tilde)
        var_loss = (d_logits ** 2).mean()
        d_eta, d_log_temp = torch.autograd.grad(
            [var_loss], [self.eta, self.log_temp], grad_outputs=torch.ones_like(var_loss))

        return {'logits': ([self.logits], [d_logits.detach()]),
                'eta': ([self.eta], [d_eta.detach()]),
                'log_temp': ([self.log_temp], [d_log_temp.detach()])
                }
