import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from config import Config
from general.functions import _Linearize
import nn_models
from methods.base import BaseNN

def Lenet(parent: BaseNN):
    class _Lenet(parent):
        def _create_module(self):
            self.in_channels = Config.DATASET_PARAMS[self.args.dataset]['in_depth']
            width = Config.DATASET_PARAMS[self.args.dataset]['width']
            height = Config.DATASET_PARAMS[self.args.dataset]['height']

            # Creating modules as attributes is necessary if we want them to have names later
            self.fc1 = self.Linear(width * height * self.in_channels, 300)
            self.fc2 = self.Linear(300, 100)
            self.fc3 = torch.nn.Linear(100, 10)

            fc = self.Sequential(
                _Linearize(),
                self.fc1,
                nn.ReLU(True),
                self.fc2,
                nn.ReLU(True),
                self.fc3
            )
            return fc
    return _Lenet