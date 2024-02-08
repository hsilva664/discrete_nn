from methods.deterministic.base import *
from methods.stochastic.composable.base import logits_to_probs_dict

class TrueGrad(Deterministic):
    def __init__(self, args, loss_obj):
        self.args = copy.deepcopy(args)
        self.loss_obj = loss_obj
        assert self.args.logits_to_probs_str is not None
        logits_to_probs = logits_to_probs_dict[args.logits_to_probs_str](self.args)
        loss_obj.mask_f = logits_to_probs.f
        self.logits_to_probs = logits_to_probs
        self.logits_to_probs.init_params()
        self.logits = self.logits_to_probs.logits
        self.logit_optim = torch.optim.SGD([self.logits], lr=args.lr)
        self.all_optim = [self.logit_optim]
        self._correct_batch_size()
        self.n = self.args.batch_size
        self.d = self.args.num_latents

    def est_str(self):
        return self.args.estimator + '_' + self.args.logits_to_probs_str

    def _prepare_iter(self):
        # Zero grads
        for optim in self.all_optim:
            optim.zero_grad()
        # Calculate true grad
        if self.args.use_true_grad:
            self.true_grad = self.loss_obj.expected_grad(self.logits)

    def _compute_grad(self):
        if self.args.use_true_grad:
            d_logits = self.true_grad
        else:
            d_logits = self.loss_obj.expected_grad(self.logits)
        return {'logits': ([self.logits], [d_logits.unsqueeze(0)])}

    def _apply_grad(self, *args, **kwargs):
        super()._apply_grad(*args, **kwargs)
        self.logits_to_probs.postprocessing()
