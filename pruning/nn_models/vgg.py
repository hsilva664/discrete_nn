import torch
import torch.nn as nn
from config import Config
from general.functions import _Linearize
from methods.base import BaseNN
from collections import OrderedDict


def VGG(parent: BaseNN):
    class _VGG(parent):
        # Block used for ResNet operations
        class _VGG_block(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, stride, padding, cdepth=3,
                         skip=False, parent_model=None, maxpool=True):
                assert parent_model is not None
                super().__init__()
                self.cdepth = cdepth

                list_conv = []
                for i in range(1, self.cdepth+1):
                    conv = parent_model.Conv(in_channels if i == 1 else out_channels, out_channels, kernel_size,
                                              stride, padding)
                    bn = nn.BatchNorm2d(out_channels)
                    relu = nn.ReLU(True)
                    list_conv += [(f'conv{i}', conv), (f'bn{i}', bn), (f'relu{i}', relu)]

                if maxpool:
                    list_conv += [('maxpool', nn.MaxPool2d(2))]
                for attr, val in list_conv:
                    setattr(self, attr, val)

                self.module = nn.Sequential(OrderedDict(list_conv))

            def forward(self, x, *args, **kwargs):
                return self.module(x)

        def _create_module(self):
            self.in_channels = Config.DATASET_PARAMS[self.args.dataset]['in_depth']
            block_1 = _VGG._VGG_block(self.in_channels, 64, 3, 1, 1, cdepth=2, parent_model=self)
            block_2 = _VGG._VGG_block(64, 128, 3, 1, 1, cdepth=2, parent_model=self)
            block_3 = _VGG._VGG_block(128, 256, 3, 1, 1, cdepth=4, parent_model=self)
            block_4 = _VGG._VGG_block(256, 512, 3, 1, 1, cdepth=4, parent_model=self)
            block_5 = _VGG._VGG_block(512, 512, 3, 1, 1, cdepth=4, parent_model=self,
                                      maxpool=False)

            self.conv_block = torch.nn.Sequential(OrderedDict(zip(["block_1", "block_2", "block_3", "block_4", "block_5"],
                                                                  [block_1, block_2, block_3, block_4, block_5])))
            width = Config.DATASET_PARAMS[self.args.dataset]['width']
            height = Config.DATASET_PARAMS[self.args.dataset]['height']
            dummy = torch.zeros((1, self.in_channels, width, height)).to(next(self.conv_block.parameters()).device)
            avg_pool = nn.AvgPool2d(2)
            out_dummy = avg_pool(self.conv_block(dummy))

            self.linear_block = self.Sequential(OrderedDict(
                [('avg_pool', avg_pool), ('linearize', _Linearize()), ('linear', nn.Linear(out_dummy.numel(), 10))]))

            return self.Sequential(self.conv_block, self.linear_block)

    return _VGG
