import torch

class Config:
    @staticmethod
    def GET_METHOD_DICT():
        from methods.base import BaseMethod
        from methods.mp import MP
        from methods.gmp import GMP
        from methods.cp import CP
        from methods.monte_carlo.loorf import LOORF
        from methods.monte_carlo.arms import ARMS
        METHOD_DICT = {
            'base': BaseMethod,
            'cp': CP,
            'mp': MP,
            'gmp': GMP,
            'loorf': LOORF,
            'arms': ARMS
        }
        return METHOD_DICT

    DATASET_PARAMS = {
        'mnist': {
            'in_depth': 1,
            'width': 28,
            'height': 28
        },
        'cifar10': {
            'in_depth': 3,
            'width': 32,
            'height': 32            
        }
    }

    @staticmethod
    def GET_NN_MODELS_TO_CLASSES_DICT():
        from nn_models.conv6 import Conv6
        from nn_models.lenet import Lenet
        from nn_models.resnet import ResNet
        from nn_models.vgg import VGG
        return {
            'conv6': Conv6,
            'lenet': Lenet,
            'resnet': ResNet,
            'vgg': VGG
            }

    @staticmethod
    def GET_BASE_NN_PARAMETERS():
        o = {}
        # CONV6
        o['conv6'] = {}
        d = o['conv6']
        d['lr'] = 3e-4
        d['epochs'] = 40
        d['wd'] = 0.0001
        d['lr_sch_pct'] = []
        d['lr_sch_mul'] = []
        d['optim'] = 'adam'
        d['val_set_size'] = 5000
        d['train_bs'] = 60
        d['val_bs'] = 500
        d['test_bs'] = 500
        d['global_prune'] = False
        d['augment'] = False
        # LENET
        o['lenet'] = {}
        d = o['lenet']
        d['lr'] = 1.2e-3
        d['epochs'] = 200
        d['wd'] = 0.0001
        d['lr_sch_pct'] = [0.4, 0.6]
        d['lr_sch_mul'] = [0.1, 0.1]
        d['optim'] = 'adam'
        d['val_set_size'] = 5000
        d['train_bs'] = 60
        d['val_bs'] = 500
        d['test_bs'] = 500
        d['global_prune'] = False
        d['augment'] = False
        # RESNET20
        o['resnet'] = {}
        d = o['resnet']
        d['lr'] = 0.1
        d['epochs'] = 200
        d['lr_sch_pct'] = [0.4, 0.6]
        d['lr_sch_mul'] = [0.1, 0.1]
        d['wd'] = 0.0001
        d['optim'] = 'sgd'
        d['momentum'] = 0.9
        d['nesterov'] = True
        d['val_set_size'] = 5000
        d['train_bs'] = 128
        d['val_bs'] = 500
        d['test_bs'] = 500 
        d['global_prune'] = True
        d['augment'] = True
        # VGG
        o['vgg'] = {}
        d = o['vgg']
        d['lr'] = 0.1
        d['epochs'] = 200
        d['lr_sch_pct'] = [0.4, 0.6]
        d['lr_sch_mul'] = [0.1, 0.1]
        d['wd'] = 1e-4
        d['optim'] = 'sgd'
        d['momentum'] = 0.9
        d['nesterov'] = True
        d['val_set_size'] = 5000
        d['train_bs'] = 64
        d['val_bs'] = 500
        d['test_bs'] = 500
        d['global_prune'] = True
        d['augment'] = True
        return o

    @staticmethod
    def get_optim(args, parameters):
        opt_str = args.optim
        if opt_str == 'adam':
            kwargs = {'weight_decay': getattr(args, 'wd', None),
                      }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            optim = torch.optim.Adam(params=parameters, lr=args.lr, **kwargs)
        elif opt_str == 'sgd':
            kwargs = {'weight_decay': getattr(args, 'wd', None),
                      'momentum': getattr(args, 'momentum', None),
                      'nesterov': getattr(args, 'nesterov', None),
                     }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            optim = torch.optim.SGD(params=parameters, lr=args.lr, **kwargs)
        elif opt_str == 'rmsprop':
            kwargs = {'weight_decay': getattr(args, 'wd', None),
                      'momentum': getattr(args, 'momentum', None),
                     }
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            optim = torch.optim.RMSprop(params=parameters, lr=args.lr, **kwargs)
        else:
            raise NotImplementedError
        return optim

    EVAL_EVERY = 1
