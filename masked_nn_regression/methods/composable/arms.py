import torch
import methods.composable.base
class ARMS(methods.composable.base.BaseMethod):

    def __init__(self, args):
        super().__init__(args)
        self.acc_A = torch.zeros(self.premask.shape, requires_grad=False, device=self.args.device)
        self.acc_B = torch.zeros(self.premask.shape, requires_grad=False, device=self.args.device)
        self.gamma = torch.distributions.gamma.Gamma(self.args.n, 1)

    def _prepare_iter(self):
        super()._prepare_iter()
        self.rho = torch.where((self.probs == 0.0) | (self.probs == 1.0), torch.tensor(0., device=self.probs.device),
                               torch.where(self.probs > 0.5, self._rho_f(self.probs), self._rho_fp(self.probs)))
        self.rho = self.rho.reshape(
            list(self.rho.shape) + [1 for _ in range(len(self.premask.shape) - len(self.rho.shape))])
        self.sum_Ei = self.gamma.sample(self.probs.shape).to(self.probs.device)
        self.all_R = self.sum_Ei.detach().clone()

    def _compute_and_apply_grad(self):
        with torch.no_grad():
            self.loss = 0.0
            x = None
            y = None
            for i in range(self.args.n):
                ip1 = i + 1
                if ip1 < self.args.n:
                    Ei = -self.all_R * torch.rand_like(self.probs) \
                         ** (1 / (self.args.n - ip1)) + self.all_R
                else:
                    Ei = self.all_R.detach().clone()
                self.all_R -= Ei
                cur_d = Ei / self.sum_Ei
                cur_u_tilde = 1 - (1 - cur_d) ** (self.args.n - 1)
                cur_eff_u = torch.where(self.probs > 0.5, cur_u_tilde, 1 - cur_u_tilde)
                z = cur_eff_u.lt(self.probs).type_as(cur_eff_u)
                d_log_prob = self.mask_f_obj.d_log_prob(z, self.premask, self.probs)
                if i == 0:
                    this_loss, x, y = self.loss_obj.tr_loss(z.unsqueeze(0), return_input=True)
                else:
                    this_loss = self.loss_obj.tr_loss(z.unsqueeze(0), x=x, y=y)
                # Update accumulators
                inner_term = d_log_prob/(1 - self.rho)
                self.acc_A.mul_(max(i - 1, 1.)/max(1., i)).add_(inner_term * this_loss/max(1., i))
                self.acc_B.mul_(max(i - 1, 1.)/max(1., i)).add_(inner_term/max(1., i))
                self.loss = (i / (i + 1)) * self.loss + (1. / (i + 1)) * this_loss

        self.premask.grad = self.acc_A - self.acc_B * self.loss
        self.acc_A.fill_(0.0)
        self.acc_B.fill_(0.0)
        self.optim.step()
        
    def _rho_f(self, probs):
        return (torch.maximum(torch.zeros_like(probs),
                              2 * (1 - probs) ** (1 / (self.args.n - 1)) - 1) ** (
                        self.args.n - 1) - (1 - probs) ** 2) / \
               (probs * (1 - probs))


    def _rho_fp(self, probs):
        return (torch.maximum(torch.zeros_like(probs),
                              2 * probs ** (1 / (self.args.n - 1)) - 1) ** (
                        self.args.n - 1) - probs ** 2) / \
               (probs * (1 - probs))

