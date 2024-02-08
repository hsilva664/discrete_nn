import torch
from aux.parser import MainParser
import methods.base
from scipy.special import logit
import numpy as np
from torch.utils.data import DataLoader

class BaseMethod(methods.base.BaseMethod):
    class Parser(MainParser):
        def __init__(self):
            super().__init__()
            self.add_argument('--n', type=int, default=10,
                               help='How many samples to use for the estimator')
            self.add_argument('--mask_f', type=str, default="sigmoid",
                               help='how to convert premask to mask', choices=list(mask_f_dict.keys()))

    def __init__(self, args):
        super().__init__(args)
        self.mask_f_obj = mask_f_dict[args.mask_f](self.args)
        self.mask_f_obj.init_params()
        self.premask = self.mask_f_obj.premask
        self.optim = self.optim_f([self.premask], self.args.lr)

    def iter(self, *args, **kwargs):
        super().iter(*args, **kwargs)
        self.mask_f_obj.postprocessing()

    def _prepare_iter(self):
        super()._prepare_iter()
        self.optim.zero_grad()
        with torch.no_grad():
            self.probs = self.mask_f_obj.f(self.premask)

    def get_val_mask(self, probs):
        THIS_BATCH_SIZE = 5 * self.loss_obj.batch_size
        THIS_ATTEMPTED_MASKS = 5
        if self.args.loss_type == 'nn_loss':
            dataset = self.loss_obj.dataset
            loader = DataLoader(dataset=dataset, batch_size=THIS_BATCH_SIZE, shuffle=True)
            x, y = next(iter(loader))
        elif self.args.loss_type == 'l0_norm_loss':
            x, y = None, None
        else:
            raise NotImplementedError
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
        probs = self.mask_f_obj.f(self.premask)
        z = self.get_val_mask(probs)
        device = probs.device
        dtype = probs.dtype
        pre_entr = -probs * torch.log(probs) -(1 - probs) * torch.log(1 - probs)
        self.norm_entr = (torch.where( (probs != 0.0) & (probs != 1.0), pre_entr, torch.tensor(0.0, device=device, dtype=dtype)).sum() /
                          (self.args.d * torch.log(torch.tensor(2.0, device=device, dtype=dtype)))).item()
        self.val_loss = self.loss_obj.val_loss(z.unsqueeze(0)).item()

    def set_state(self, source_dict):
        self.lr = source_dict['lr']
        self.premask.data = source_dict['premask_data']
        self.optim.load_state_dict(source_dict['optim_state'])

    def get_state(self, target_dict):
        method_dict = {
            'lr': self.lr,
            'premask_data': self.premask.data,
            'optim_state': self.optim.state_dict()
        }
        target_dict.update(method_dict)

class ComposablePremaskToMask:
    def __init__(self, args):
        self.args = args
        self.premask = None

    def init_params(self):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def d_log_prob(self, z, premask, probs):
        raise NotImplementedError

    def postprocessing(self):
        pass

class Sigmoid(ComposablePremaskToMask):
    def init_params(self):
        self.premask = torch.zeros(self.args.d, requires_grad=True, device=self.args.device)
        torch.nn.init.constant_(self.premask, logit(self.args.mask_init))

    def f(self, x):
        return torch.sigmoid(x)

    def d_log_prob(self, z, premask, probs):
        with torch.no_grad():
            return z - probs

class Escort(ComposablePremaskToMask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = 4

    def init_params(self):
        self.premask = torch.zeros((self.args.d, 2), requires_grad=True, device=self.args.device)
        prm0, prm1 = self._get_escort_init()
        torch.nn.init.constant_(self.premask[:, 0], prm0)
        torch.nn.init.constant_(self.premask[:, 1], prm1)

    def _get_escort_init(self):
        iv = self.args.mask_init
        prm0 = 1.
        prm1 = prm0 * (iv / (1 - iv)) ** (1 / self.p)
        return prm0, prm1

    def f(self, x):
        p = self.p
        f1 = torch.abs(x[..., 1]) ** p
        f0 = torch.abs(x[..., 0]) ** p
        return f1 / (f1 + f0)

    def d_log_prob(self, z, premask, probs):
        with torch.no_grad():
            p = self.p
            nonzero_d_log_prob = torch.exp((torch.log(p * torch.abs(probs - z).unsqueeze(-1))
                                - torch.log(torch.abs(premask))))
            nonzero_d_log_prob *= torch.sign(probs - z).unsqueeze(-1) * torch.sign(premask)
            out = torch.where(premask == 0.0, torch.tensor(0.0).to(premask.device), nonzero_d_log_prob)
            out[..., 1] *= -1
        return out

class Cos(ComposablePremaskToMask):
    def init_params(self):
        self.premask = torch.zeros(self.args.d, requires_grad=True, device=self.args.device)
        torch.nn.init.constant_(self.premask, self._cosine_mask_inv(torch.tensor(self.args.mask_init)))

    def f(self, x):
        return 0.5 * (1 - torch.cos(x))

    def _cosine_mask_inv(self, y):
        # The function is not invertible, but we restrict it to the inverval [0,Pi]
        return -torch.arccos(2 * y - 1) + np.pi

    def d_log_prob(self, z, premask, probs):
        with torch.no_grad():
            z_to_sign = 2 * z - 1
            nonzero_d_log_prob = z_to_sign * torch.sin(premask)/(1 - z_to_sign * torch.cos(premask))
            out = torch.where((probs == 0.0) | (probs == 1.0), torch.tensor(0.0).to(premask.device), nonzero_d_log_prob)
        return out

class Direct(ComposablePremaskToMask):
    def init_params(self):
        self.premask = torch.zeros(self.args.d, requires_grad=True, device=self.args.device)
        torch.nn.init.constant_(self.premask, self.args.mask_init)

    def f(self, x):
        return x

    def d_log_prob(self, z, premask, probs):
        with torch.no_grad():
            nonzero_d_log_prob = (z - probs)/(probs*(1-probs))
            out = torch.where(probs.eq(0.0) | probs.eq(1.0),
                              torch.tensor(0.0).to(probs.device), nonzero_d_log_prob)
        return out

    def postprocessing(self, eps=1E-6):
        with torch.no_grad():
            self.premask.data.clamp_(eps, 1.0 - eps)

mask_f_dict = {
    'sigmoid': Sigmoid,
    'cos': Cos,
    'escort': Escort,
    'direct': Direct,
}