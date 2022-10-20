import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset


from models_resnet import Resnet18_md, Resnet50_md
from data_loader import KittiLoader
from transforms import image_transforms


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError(f"Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels=3, pretrained=False):
    if model == 'resnet50_md':
        out_model = Resnet50_md(input_channels)
    elif model == 'resnet18_md':
        out_model = Resnet18_md(input_channels)
    else:
        out_model = ResnetModel(input_channels, encoder=model, pretrained=pretrained)
    return out_model


def prepare_dataloader(data_directory, mode, filenames, augment_parameters,
                       do_augmentation, batch_size, size, num_workers):
    data_dirs = os.listdir(data_directory)
    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size=size)
    datasets = [KittiLoader(data_directory, mode, filenames, transform=data_transform)]
    dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    print('mode:', mode, ': Use a dataset with ', n_img, 'images')
    if mode == 'train':
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    return n_img, loader


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
