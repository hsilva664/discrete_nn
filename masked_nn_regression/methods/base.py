import torch
import functools

opt_dict = {
    'sgd': functools.partial(torch.optim.SGD, nesterov=True, momentum=0.9),
    'adam': torch.optim.Adam,
    'rmsprop': torch.optim.RMSprop
}

class BaseMethod:

    def __init__(self, args):
        self.loss_obj = args.loss_obj
        self.args = args
        self.epoch = None
        self.epoch_i = None
        self.global_i = None
        self.lr = self.args.lr
        self.optim_f = opt_dict[self.args.optim_str]

    def iter(self, epoch, epoch_i, global_i):
        self.epoch = epoch
        self.epoch_i = epoch_i
        self.global_i = global_i
        self._prepare_iter()
        self._compute_and_apply_grad()

    def _prepare_iter(self):
        if self.epoch_i == 0:
            self._adjust_lr()

    def _compute_and_apply_grad(self):
        pass

    def val_iter(self):
        pass

    def _adjust_lr(self):
        if self.epoch in self.args.lr_sch_epochs and self.epoch_i == 0:
            mul = self.args.lr_sch_mul[ self.args.lr_sch_epochs.index(self.epoch) ]
            self.lr = self.lr * mul
            for param_group in self.optim.param_groups: param_group['lr'] = param_group['lr'] * mul

    def set_state(self, source_dict):
        raise NotImplementedError

    def get_state(self, target_dict):
        raise NotImplementedError
