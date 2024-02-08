import copy

import torch
from torch.utils.data import DataLoader
import itertools
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class L0NormLoss:
    def __init__(self, args):
        self.args = args
        self.d = 8050
        self.batch_size = NNLoss.batch_size
        self.iters_per_epoch = int(np.ceil(NNLoss.dataset_size / NNLoss.batch_size))

    def tr_loss(self, mask, return_input=None, **kwargs):
        all_losses = mask.mean(dim=1)
        if return_input:
            return all_losses, None, None
        else:
            return all_losses

    def val_loss(self, mask):
        assert mask.shape[0] == 1
        return mask.mean()

# ----------- NN loss
class NNLoss:
    data_size = 10
    dataset_size = 10000
    batch_size = 100
    val_dataset_size = 5000
    val_batch_size = 5000

    def __init__(self, args):
        self.args = args  # number of hidden layers (does not count input and output layers)
        self.iters_per_epoch = int(np.ceil(self.dataset_size / self.batch_size))
        self.distance = torch.nn.PairwiseDistance(p=2.0)
        self.nn = _NNLossNet(data_size=self.data_size).to(self.args.device)
        self.dataset = _NNDataset(data_size=self.data_size, dataset_size=self.dataset_size, device=self.args.device,
                                  main_loss_obj=self)
        self.val_dataset = _NNDataset(data_size=self.data_size, dataset_size=self.val_dataset_size,
                                      device=self.args.device, main_loss_obj=self)
        self.loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.val_batch_size, shuffle=False)

        self.loader_generator_obj = self.loader_generator_f()
        self._d = None

    @property
    def d(self):
        if self._d is None:
            self._d = sum([w.numel() for w in self.nn.weights])
        return self._d

    def loader_generator_f(self):
        while True:
            for input, target in self.loader:
                yield input, target

    def tr_loss(self, mask, x=None, y=None, return_input=False, force_eval_mode=False):
        if force_eval_mode:
            self.nn.eval()
        else:
            self.nn.train()
        if x is None or y is None:
            x, y = next(self.loader_generator_obj)
        all_losses = []
        for i in range(mask.shape[0]):
            all_losses.append(torch.mean(self.distance(self.nn(mask[i], x), y), dim=0))
        o = torch.stack(all_losses, dim=0)
        if return_input:
            return o, x, y
        else:
            return o

    def val_loss(self, mask):
        self.nn.eval()
        assert mask.shape[0] == 1
        if self.args.visualize:
            assert self.data_size == 1
            plt.clf()
            for inp, target in self.val_loader:
                plt.scatter(inp.view(-1).cpu().numpy(), target.view(-1).cpu().numpy(), s=.01, c="red")

        val_loss = 0.
        sz = 0
        with torch.no_grad():
            for inp, target in self.val_loader:
                sz += inp.shape[0]
                pred = self.nn(mask[0], inp)
                if self.args.visualize:
                    plt.scatter(inp.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy(), s=.01)
                val_loss = val_loss + torch.sum(self.distance(pred, target), dim=0) / float(
                     self.val_dataset_size)
            if self.args.visualize:
                plt.show()
            assert self.val_dataset_size == sz
            return val_loss


class _NNLossNet(torch.nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.data_size = data_size
        self.hidden_sizes = [50, 50, 50, 50]
        all_sizes = [data_size] + self.hidden_sizes + [1]
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(all_sizes[sz_i], all_sizes[sz_i-1]), requires_grad=False)
             for sz_i in range(1, len(all_sizes))]
        )
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(self.hidden_sizes[sz_i], affine=False, track_running_stats=False)
                for sz_i in range(len(self.hidden_sizes))])
        for i in range(len(self.weights)):
            torch.nn.init.xavier_normal_(self.weights[i])
        self.f = [F.leaky_relu for _ in range(len(self.weights) - 1)]

    def forward(self, mask, x):
        base_index = 0
        f_index = 0
        for i in range(len(self.weights)):
            f_index = base_index + self.weights[i].numel()
            w = mask[base_index:f_index].reshape_as(self.weights[i]) * self.weights[i]
            x = F.linear(x, w, bias=None)
            if i != (len(self.weights) - 1):
                x = self.bns[i](x)
                x = self.f[i](x)
            base_index = f_index
        assert f_index == mask.numel()
        return x


class _NNDataset(torch.utils.data.Dataset):
    def __init__(self, data_size, dataset_size, device, main_loss_obj):
        self.data_size = data_size
        self.dataset_size = dataset_size
        self.dataset_device = device
        self.main_loss_obj = main_loss_obj
        self.i_range = 1.  # input goes from -i_range to i_range
        self.target_f = OtherNNTargetF(self)
        self.input_data = self.target_f.init_x()
        self.target_data = self.target_f(self.input_data)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index: int):
        return self.input_data[index], self.target_data[index]

# ----------- Target functions to use with NN loss


class BaseF:
    def __init__(self, parent_dataset):
        self.parent_dataset = parent_dataset

    def init_x(self):
        return (2 * self.parent_dataset.i_range) * \
            torch.rand(self.parent_dataset.dataset_size, self.parent_dataset.data_size,
                       device=self.parent_dataset.dataset_device) - self.parent_dataset.i_range

# ----------- Targets with one dimensional inputs
class Base1dF(BaseF):
    def __init__(self, parent_dataset):
        super().__init__(parent_dataset)
        assert parent_dataset.data_size == 1

class SigTarget1dF(Base1dF):
    def __call__(self, x: torch.Tensor):
        return torch.sigmoid(-(x + 1.) / 2.)

class ConstantTarget1dF(Base1dF):
    def __call__(self, x: torch.Tensor):
        return 0.3 * torch.ones_like(x)

class NormalizedDSig1dF(Base1dF):
    def __call__(self, x: torch.Tensor):
        # normalized to be in [0,1] range
        return 4.0 * (torch.sigmoid(x) * (1 - torch.sigmoid(x)))


class NormalizedSin1dF(Base1dF):
    def __init__(self, *args, freq=1.0, yrange=(0.0, 1.0), **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        self.yrange = yrange
    def __call__(self, x: torch.Tensor):
        # normalized to be in correct range
        return (self.yrange[1] - self.yrange[0])*((torch.sin(self.freq * x) - (-1)) / (1 - (-1))) + self.yrange[0]


class MultimodalTarget1dF(Base1dF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalized_dsig = NormalizedDSig1dF(parent_dataset=self.parent_dataset)
        self.normalized_sin = NormalizedSin1dF(parent_dataset=self.parent_dataset)

    def __call__(self, x: torch.Tensor):
        return 1 - 2 * self.normalized_dsig(x / (0.3 * self.parent_dataset.i_range)) * \
            self.normalized_sin(((x / (0.85 * self.parent_dataset.i_range)) ** 4) * \
                                (x / (0.1 * self.parent_dataset.i_range)))

# -----------  Targets with multi-dimensional inputs
class SameNNTargetF(BaseF):
    def __init__(self, *args, nn=None, soft_target=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.parent_dataset.main_loss_obj, 'dataset'):
            # This is the train dataset
            self.NN = copy.deepcopy(nn)
            weights = torch.cat([w.view(-1) for w in self.NN.weights.parameters()])
            if soft_target:
                self.mask = torch.rand_like(weights)
            else:
                self.mask = torch.rand_like(weights).lt(0.5).type_as(weights)
        else:
            # This is the val dataset
            self.NN = self.parent_dataset.main_loss_obj.dataset.target_f.NN
            self.mask = self.parent_dataset.main_loss_obj.dataset.target_f.mask

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            return self.NN(self.mask, x)

class OtherNNTargetF(BaseF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self.parent_dataset.main_loss_obj, 'dataset'):
            # This is the train dataset
            self.NN = _TargetNN(self.parent_dataset.data_size).to(self.parent_dataset.main_loss_obj.args.device)
        else:
            # This is the val dataset
            self.NN = self.parent_dataset.main_loss_obj.dataset.target_f.NN

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            return self.NN(x)


class _TargetNN(torch.nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.output_range = None
        self.data_size = data_size
        self.hidden_sizes = [500, 500, 500, 500, 500]
        all_sizes = [data_size] + self.hidden_sizes + [1]
        self.weights = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.Tensor(all_sizes[sz_i], all_sizes[sz_i-1]), requires_grad=False)
             for sz_i in range(1, len(all_sizes))]
        )
        for i in range(len(self.weights)):
            self.weights[i].data = 2*torch.bernoulli(self.weights[i].data, 0.5) - 1
        self.bns = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(self.hidden_sizes[sz_i], affine=False, track_running_stats=True, momentum=1.0)
                for sz_i in range(len(self.hidden_sizes))])
        self.f = [F.leaky_relu for _ in range(len(self.weights) - 1)]
        self.train()

    def forward(self, x):
        for i in range(len(self.weights)):
            w = self.weights[i]
            x = F.linear(x, w, bias=None)
            if i != (len(self.weights) - 1):
                x = self.bns[i](x)
                x = self.f[i](x)
            else:
                # normalize only after you have all the data
                pass
        if self.output_range is None:
            self.output_range = (x.min(), x.max())
            self.eval()
        return (x - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
