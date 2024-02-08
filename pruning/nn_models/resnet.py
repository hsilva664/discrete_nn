import torch.nn as nn
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from config import Config
from general.functions import _Linearize
from methods.base import BaseNN


def ResNet(parent: BaseNN):
    class _ResNet(parent):
        # Block used for ResNet operations
        class _Res_block(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, padding, cdepth=3, i_stride=False,
                         skip=False, parent_model=None):
                assert parent_model is not None
                super().__init__()
                self.cdepth = cdepth
                if i_stride:
                    i_stride = 2
                else:
                    i_stride = stride

                # The output of each of these sub-blocks gets added to the skip convolution
                for i in range(self.cdepth):
                    conv1 = parent_model.Conv(in_channels if i == 0 else out_channels, out_channels, kernel_size,
                                              i_stride if i == 0 else stride, padding)
                    bn1 = nn.BatchNorm2d(out_channels)
                    relu = nn.ReLU(True)
                    conv2 = parent_model.Conv(out_channels, out_channels, kernel_size, stride, padding)
                    bn2 = nn.BatchNorm2d(out_channels)
                    curr_sub_block = parent_model.Sequential(
                        OrderedDict([('conv1', conv1), ('bn1', bn1), ('relu', relu), ('conv2', conv2), ('bn2', bn2)]))
                    setattr(self, f'sub_block_{i + 1}', curr_sub_block)

                self.skip = skip
                if self.skip:
                    self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, i_stride, 0, bias=False)

            def forward(self, x, *args, **kwargs):
                for i in range(self.cdepth):
                    r = self.skip_conv(x) if self.skip and i == 0 else x
                    x = getattr(self, f'sub_block_{i + 1}')(x, *args, **kwargs)
                    x = F.relu(x + r, inplace=True)
                return x

        def _create_module(self):
            self.in_channels = Config.DATASET_PARAMS[self.args.dataset]['in_depth']

            self.input_block = self.Sequential(OrderedDict(
                [('conv', self.Conv(self.in_channels, 16, 3, 1, 1)), ('bn', nn.BatchNorm2d(16)),
                    ('relu', nn.ReLU(True))]))

            self.block_1 = _ResNet._Res_block(16, 16, 3, 1, 1, parent_model=self)
            self.block_2 = _ResNet._Res_block(16, 32, 3, 1, 1, i_stride=True, skip=True, parent_model=self)
            self.block_3 = _ResNet._Res_block(32, 64, 3, 1, 1, i_stride=True, skip=True, parent_model=self)

            self.linear_block = self.Sequential(OrderedDict(
                [('avg_pool', nn.AvgPool2d(8)), ('linearize', _Linearize()), ('linear', nn.Linear(64, 10))]))

            return self.Sequential(self.input_block, self.block_1, self.block_2, self.block_3, self.linear_block)

    return _ResNet
