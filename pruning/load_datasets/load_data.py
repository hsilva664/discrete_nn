from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import functools

def generate_loaders(args):
    dataset_name = args.dataset
    # Preprocess images
    augment = args.augment
    train_transform = None
    test_transform = None

    if dataset_name == 'cifar10':
        if augment:
            mean, std = [x / 255 for x in [125.3, 123.0, 113.9]],  [x / 255 for x in [63.0, 62.1, 66.7]]
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:
            # No augmentation usually means Conv6, we use a different normalization for it to reproduce Frankle's results
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        d_fun = datasets.CIFAR10
        n_classes = 10
    elif dataset_name == 'mnist':
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = train_transform
        d_fun = datasets.MNIST
        n_classes = 10

    addr = './data/' + dataset_name
    train_dataset = d_fun(addr, train=True, download=True, transform=train_transform, target_transform=torch.tensor)
    val_dataset = d_fun(addr, train=True, download=True, transform=test_transform, target_transform=torch.tensor)
    tgt = torch.tensor(train_dataset.targets) if not isinstance(train_dataset.targets, torch.Tensor) else train_dataset.targets
    label_dict_v, label_dict_idxs = torch.sort(tgt)
    n_ex_per_class = torch.cat([torch.tensor([0]), torch.cumsum(torch.bincount(label_dict_v),0)])
    label_dict = list(map(lambda p: (label_dict_idxs[n_ex_per_class[p].item() : n_ex_per_class[p+1].item() ]).numpy(), range(len(train_dataset.classes)) ))

    train_indices = []
    val_indices = []

    val_set_size = args.val_set_size
    train_bs = args.train_bs
    val_bs = args.val_bs
    test_bs = args.test_bs
    n_workers = args.n_workers

    # Shuffle and balance
    for idxs in label_dict:
        np.random.shuffle(idxs)
        train_indices.append(idxs[(val_set_size//n_classes):])
        val_indices.append(idxs[:(val_set_size//n_classes)])

    train_indices = np.concatenate( train_indices)
    val_indices = np.concatenate( val_indices)

    test_dataset = d_fun(addr, train=False, download=True, transform=test_transform, target_transform=torch.tensor)
    args.test_set_size = len(test_dataset)
    assert val_set_size < len(train_dataset)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DeviceDataloader(args, train_dataset, shuffle=False, sampler=train_sampler,
        batch_size=train_bs, num_workers=n_workers, pin_memory=True)
    val_loader = DeviceDataloader(args, val_dataset, shuffle=False, sampler=valid_sampler,
        batch_size=val_bs, num_workers=n_workers, pin_memory=True)
    test_loader = DeviceDataloader(args, test_dataset, shuffle=False,
        batch_size=test_bs, num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


class DeviceDataloader(torch.utils.data.DataLoader):
    def __init__(self, run_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_args = run_args
    def __iter__(self):
        it = super().__iter__()
        return _DeviceDataloaderIter(it, self.run_args)

class _DeviceDataloaderIter:
    def __init__(self, it, run_args):
        self.it = it
        self.run_args = run_args
    def __next__(self):
        data, target = next(self.it)
        return data.to(self.run_args.device), target.to(self.run_args.device)
