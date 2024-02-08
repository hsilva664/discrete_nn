import torch
import torch.nn as nn
from config import Config
from general.functions import _Linearize
from methods.base import BaseNN
from collections import OrderedDict


def Conv6(parent: BaseNN):
    class _Conv6(parent):
        def _create_module(self):
            self.in_channels = Config.DATASET_PARAMS[self.args.dataset]['in_depth']
            self.depths = [64, 128, 256]
            self.conv_block = self.Sequential(OrderedDict(
                [("conv_1", self.Conv(self.in_channels, self.depths[0], 3, stride=1, padding=1)),
                    ("relu_1", nn.ReLU(True)),
                    ("conv_2", self.Conv(self.depths[0], self.depths[0], 3, stride=1, padding=1)),
                    ("maxpool_1", nn.MaxPool2d(2, 2)), ("relu_2", nn.ReLU(True)),
                    ("conv_3", self.Conv(self.depths[0], self.depths[1], 3, stride=1, padding=1)),
                    ("relu_3", nn.ReLU(True)),
                    ("conv_4", self.Conv(self.depths[1], self.depths[1], 3, stride=1, padding=1)),
                    ("maxpool_2", nn.MaxPool2d(2, 2)), ("relu_4", nn.ReLU(True)),
                    ("conv_5", self.Conv(self.depths[1], self.depths[2], 3, stride=1, padding=1)),
                    ("relu_5", nn.ReLU(True)),
                    ("conv_6", self.Conv(self.depths[2], self.depths[2], 3, stride=1, padding=1)),
                    ("maxpool_3", nn.MaxPool2d(2, 2))]))

            in_depth = Config.DATASET_PARAMS[self.args.dataset]['in_depth']
            width = Config.DATASET_PARAMS[self.args.dataset]['width']
            height = Config.DATASET_PARAMS[self.args.dataset]['height']
            dummy = torch.zeros((1, in_depth, width, height)).to(next(self.conv_block.parameters()).device)

            out_dummy = self.conv_block(dummy)

            self.fc_block = self.Sequential(OrderedDict([('linearize', _Linearize()), ('relu_1', nn.ReLU(True)),
                ('linear_1', self.Linear(out_dummy.numel(), 256)), ('relu_2', nn.ReLU(True)),
                ('linear_2', self.Linear(256, 256)), ('relu_3', nn.ReLU(True)),
                ('linear_final', nn.Linear(256, 10)), ]))

            return self.Sequential(self.conv_block, self.fc_block)

    return _Conv6
