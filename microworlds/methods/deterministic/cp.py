from methods.deterministic.base import *

class CP(Deterministic):

    def __init__(self, args, loss_obj):
        self.args = copy.deepcopy(args)
        loss_obj.mask_f = self._cp_hard_mask
        # Initialize so that the prob is 0.5
        self.logits = torch.zeros(args.num_latents, requires_grad=True)
        # Temperature schedule
        self.temp = 1.
        self.pseudo_epoch_iters = 100.
        n_pseudo_epochs = np.floor(self.args.iters / self.pseudo_epoch_iters)
        self.delta_temp = self.args.cp_final_temp ** (1. / n_pseudo_epochs)
        # lr schedule
        self.lr_decay_iters = np.array(self.args.cp_lr_sch_pct) * self.args.iters
        # Same as parent class
        self.loss_obj = loss_obj
        self.logit_optim = torch.optim.SGD([self.logits], lr=args.lr)
        self.all_optim = [self.logit_optim]
        self._correct_batch_size()

    def _cp_hard_mask(self, premask):
        return (premask.gt(0.)).type_as(premask)

    def _cp_soft_mask(self, premask):
        return torch.sigmoid(self.temp * premask)

    def iter(self, iteration):
        if iteration % int(self.pseudo_epoch_iters) == 0:
            self.temp = np.minimum(self.temp * self.delta_temp, self.args.cp_final_temp)
        if iteration in self.lr_decay_iters:
            m = self.args.cp_lr_sch_mul.pop(0)
            for param_group in self.logit_optim.param_groups: param_group['lr'] = param_group['lr'] * m
        super().iter(iteration)

    def _compute_grad(self):
        if self.args.use_true_grad:
            raise 'This should not be used with true grad at this point (2023/02)'
            # d_logits = self.true_grad
        else:
            mask = self._cp_soft_mask(self.logits)
            sig_loss = self.loss_obj.sample_loss(mask.unsqueeze(0))
            d_logits = torch.autograd.grad(
                [sig_loss], [self.logits], grad_outputs=torch.ones_like(sig_loss))[0]
        return {'logits': ([self.logits], [d_logits.unsqueeze(0)])}

    def _compute_expected_loss_for_logger(self):
        thetas = self.logits.detach().gt(0.).type_as(self.logits)
        return self.loss_obj.sample_loss(thetas.unsqueeze(0), is_train=False).item()