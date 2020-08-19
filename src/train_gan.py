import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import time
from PIL import Image

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import seaborn as sns
import glob
import configparser

from gan import GAN_Img_Dataset
from gan import ImageTransform

from utils import read_config


def read_data():
    config_ini = read_config.read_config()
    train_path = config_ini.get('PATH', 'train_data')
    return glob.glob(train_path)


def make_dataset(data_list):
    # parameters ------------------------------------------
    config_ini = read_config.read_config()
    mean = eval(config_ini.get('DATA', 'mean'))
    std = eval(config_ini.get('DATA', 'std'))
    batch_size = config_ini.getint('DATA', 'batch_size')
    shuffle = config_ini.getboolean('DATA', 'shuffle')
    # -----------------------------------------------------
    train_dataset = GAN_Img_Dataset(file_list=data_list, transform=ImageTransform(mean, std))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader

def weights_init(m):
    """
    This function initialize Conv2d, ConvTranspose2d and BatchNorm2d.
    :param model: generator or discriminator
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv2dとConvTranspose2dの初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm2dの初期化
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# test -----------------------------------------------------------------------------------------------------------------
# def check_dataloader():
#     data = read_data()
#     data_loader = make_dataset(data)
#     batch_iterator = iter(data_loader)
#     imges = next(batch_iterator)
#     print(imges.size())
#
# check_dataloader()