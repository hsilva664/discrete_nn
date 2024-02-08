from losses.base_loss import BaseLoss
import torch

class MSELoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.target = torch.Tensor(1, args.num_latents)
        self.target.fill_(args.target)

    def sample_loss(self, sample, is_train=True):
        with torch.set_grad_enabled(is_train):
            return ((sample - self.target) ** 2).mean(1)
