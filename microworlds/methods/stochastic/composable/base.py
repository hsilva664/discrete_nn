import torch
import numpy as np
from methods.stochastic.base import *
import copy

class Composable(Stochastic):
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
        raise NotImplementedError
    def _compute_grad(self):
        raise NotImplementedError

    def _apply_grad(self, *args, **kwargs):
        super()._apply_grad(*args, **kwargs)
        self.logits_to_probs.postprocessing()


class ComposableLogitsToProbs:
    def __init__(self, args):
        self.args = args
        self.logits = None

    def init_params(self):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def d_log_prob(self, z, prob_z, logits, probs):
        raise NotImplementedError

    def postprocessing(self):
        pass

class Sigmoid(ComposableLogitsToProbs):
    def init_params(self):
        self.logits = torch.zeros(self.args.num_latents, requires_grad=True)

    def f(self, x):
        return torch.sigmoid(x)

    def d_log_prob(self, z, prob_z, logits, probs):
        with torch.no_grad():
            return z - probs

class Escort(ComposableLogitsToProbs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = self.args.escort_p

    def init_params(self):
        self.logits = torch.zeros((self.args.num_latents, 2), requires_grad=True)
        prm0, prm1 = self._get_escort_init(iv=0.5)
        torch.nn.init.constant_(self.logits[:, 0], prm0)
        torch.nn.init.constant_(self.logits[:, 1], prm1)

    def _get_escort_init(self, iv):
        prm0 = 1.
        prm1 = prm0 * (iv / (1 - iv)) ** (1 / self.p)
        return prm0, prm1

    def f(self, x):
        p = self.p
        f1 = torch.abs(x[..., 1]) ** p
        f0 = torch.abs(x[..., 0]) ** p
        return f1 / (f1 + f0)

    def d_log_prob(self, z, prob_z, logits, probs):
        with torch.no_grad():
            p = self.p
            nonzero_d_log_prob = torch.exp((torch.log(p * torch.abs(probs - z).unsqueeze(-1))
                                - torch.log(torch.abs(logits))))
            nonzero_d_log_prob *= torch.sign(probs - z).unsqueeze(-1) * torch.sign(logits)
            out = torch.where(logits == 0.0, torch.tensor(0.0).to(logits.device), nonzero_d_log_prob)
            out[..., 1] *= -1
        return out

class Cos(ComposableLogitsToProbs):
    def init_params(self):
        self.logits = torch.zeros(self.args.num_latents, requires_grad=True)
        torch.nn.init.constant_(self.logits, self._cosine_mask_inv(torch.tensor(0.5)))

    def f(self, x):
        return 0.5 * (1 - torch.cos(x))

    def _cosine_mask_inv(self, y):
        # The function is not invertible, but we restrict it to the inverval [0,Pi]
        return -torch.arccos(2 * y - 1) + np.pi

    def d_log_prob(self, z, prob_z, logits, probs):
        with torch.no_grad():
            z_to_sign = 2 * z - 1
            nonzero_d_log_prob = z_to_sign * torch.sin(logits)/(1 - z_to_sign * torch.cos(logits))
            out = torch.where((probs == 0.0) | (probs == 1.0), torch.tensor(0.0).to(logits.device), nonzero_d_log_prob)
        return out

class Direct(ComposableLogitsToProbs):
    def init_params(self):
        self.logits = torch.zeros(self.args.num_latents, requires_grad=True)
        torch.nn.init.constant_(self.logits, 0.5)

    def f(self, x):
        return x

    def d_log_prob(self, z, prob_z, logits, probs):
        with torch.no_grad():
            nonzero_d_log_prob = (z - probs)/(probs*(1-probs))
            out = torch.where(probs.eq(0.0) | probs.eq(1.0),
                              torch.tensor(0.0).to(probs.device), nonzero_d_log_prob)
        return out

    def postprocessing(self):
        with torch.no_grad():
            self.logits.data.clamp_(0.0, 1.0)

logits_to_probs_dict = {
    "sigmoid": Sigmoid,
    "escort": Escort,
    "cos": Cos,
    "direct": Direct
}
