import torch
import methods.composable.base
class REINFORCE(methods.composable.base.BaseMethod):

    def __init__(self, args):
        super().__init__(args)
        self.acc_A = torch.zeros(self.premask.shape, requires_grad=False, device=self.args.device)

    def _prepare_iter(self):
        super()._prepare_iter()

    def _compute_and_apply_grad(self):
        with torch.no_grad():
            self.loss = 0.0
            x = None
            y = None
            for i in range(self.args.n):
                raw_u = torch.rand_like(self.probs)
                z = raw_u.lt(self.probs).type_as(self.probs)
                d_log_prob = self.mask_f_obj.d_log_prob(z, self.premask, self.probs)
                if i == 0:
                    this_loss, x, y = self.loss_obj.tr_loss(z.unsqueeze(0), return_input=True)
                else:
                    this_loss = self.loss_obj.tr_loss(z.unsqueeze(0), x=x, y=y)
                # Update accumulators
                self.acc_A.mul_(i/(i + 1.)).add_(d_log_prob * this_loss/(i + 1.))
                self.loss = (i / (i + 1)) * self.loss + (1. / (i + 1)) * this_loss

        self.premask.grad = self.acc_A.detach().clone()
        self.acc_A.fill_(0.0)
        self.optim.step()

