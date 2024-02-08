from losses.base_loss import BaseLoss
import torch
import numpy as np
import os



class NNLossNet(torch.nn.Module):
    def __init__(self, num_latents, hidden_size=20, nlayers=10):
        super().__init__()
        l_list = [torch.nn.Sequential(torch.nn.Linear(num_latents if i == 0 else hidden_size, hidden_size, bias=False),
                                      torch.nn.BatchNorm1d(hidden_size, affine=False),
                                      torch.nn.LeakyReLU()) for i in range(nlayers)] + \
                 [torch.nn.Linear(hidden_size, 1), torch.nn.BatchNorm1d(1, affine=False)]
        self.all_l = torch.nn.Sequential(*l_list)
        for m in self.all_l.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = 2*torch.bernoulli(m.weight.data, 0.5) - 1
        for w in self.all_l.parameters(recurse=True):
            w.requires_grad = False
        # Init batchnorm
        init_bn_iters = 1000
        init_bn_bs = 100
        self.train()
        for _ in range(init_bn_iters):
            init_x = torch.rand((init_bn_bs, num_latents))
            self(init_x)
        self.eval()

    def forward(self, z):
        z = z * 2. - 1.
        return self.all_l(z)

class NNLoss(BaseLoss):
    def __init__(self, args):
        super().__init__(args)
        self.nn_model = NNLossNet(args.num_latents)

    def sample_loss(self, sample, is_train=True):
        with torch.set_grad_enabled(is_train):
            self.nn_model.eval()  # dont e.g. update bn here
            return self.nn_model(sample).squeeze(1)
