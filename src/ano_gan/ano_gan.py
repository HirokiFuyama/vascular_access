import torch
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.utils import read_config
from src.ano_gan.gan import GAN_Img_Dataset, ImageTransform


def read_model():
    """
    """
    # parameters-----------------------------------
    config_ini = read_config.read_config()
    G = config_ini.get('PATH', 'g_save')
    D = config_ini.get('PATH', 'd_save')
    # ---------------------------------------------
    return G, D


def read_data(path_type=True):
    """
    :param path_type: noise dir is True, normal dir is False
    :return:
    """
    config_ini = read_config.read_config()
    if path_type is True:
        # parameters----------------------------------------------
        path = config_ini.get('PATH', 'test_noise_images')
    else:
        # parameters----------------------------------------------
        path = config_ini.get('PATH', 'test_normal_images')
    return glob.glob(path)


def make_test_dataset(path_type=True):
    """
    """
    # parameters------------------------------------------
    config_ini = read_config.read_config()
    mean = eval(config_ini.get('DATALOADER', 'mean'))
    std = eval(config_ini.get('DATALOADER', 'std'))
    shuffle = config_ini.getboolean('DATALOADER', 'shuffle')
    # -----------------------------------------------------
    if path_type is True:
        # parameters----------------------------------------------
        batch_size = config_ini.getint('DATALOADER', 'batch_noise')
    else:
        # parameters----------------------------------------------
        batch_size = config_ini.getint('DATALOADER', 'batch_normal')

    data_list = read_data()
    test_dataset = GAN_Img_Dataset(file_list=data_list, transform=ImageTransform(mean, std))
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_dataset


def anomaly_score(real_image, fake_img, D):
    """
    """
    # parameters-----------------------------------
    config_ini = read_config.read_config()
    lam = config_ini.getfloat('ANOMALY', 'lambda')
    # ---------------------------------------------

    # pixel difference
    residual_loss = torch.abs(real_image - fake_img)
    residual_loss = residual_loss.view(residual_loss.size()[0], -1)
    residual_loss = torch.sum(residual_loss, dim=1)

    # fetch discriminator feature
    _, real_feature = D(real_image)
    _, G_feature = D(fake_img)

    # feature difference
    discrimination_loss = torch.abs(real_feature-G_feature)
    discrimination_loss = discrimination_loss.view(discrimination_loss.size()[0], -1)
    discrimination_loss = torch.sum(discrimination_loss, dim=1)

    # calculate anomaly score
    loss_each = (1-lam)*residual_loss + lam*discrimination_loss
    total_loss = torch.sum(loss_each)

    return total_loss, loss_each


def optimize_z(path_type=True):
    """
    """
    # parameters----------------------------------------------
    config_ini = read_config.read_config()
    z_dim = config_ini.getint('Z', 'z_dim')
    z_lr = config_ini.getfloat('Z', 'z_lr')
    epoch = config_ini.getint('Z', 'epoch')
    if path_type is True:
        batch_size = config_ini.getint('DATALOADER', 'batch_noise')
    else:
        batch_size = config_ini.getint('DATALOADER', 'batch_normal')
    # ----------------------------------------------------------

    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device：", device)

    # dataset to iterator
    test_dataset = make_test_dataset(path_type)
    batch_iterator = iter(test_dataset)
    test_images = next(batch_iterator)
    test_images = test_images.to(device)

    # define z
    z = torch.randn(batch_size, z_dim).to(device)
    z = z.view(z.size(0), z.size(1), 1, 1)
    z.requires_grad = True
    z_optimizer = torch.optim.Adam([z], lr=z_lr)

    G, D = read_model()

    for i in range(epoch):
        fake_img = G(z)
        loss, _ = anomaly_score(test_images, fake_img, D)

        z_optimizer.zero_grad()
        loss.backward()
        z_optimizer.step()

        if epoch % 1000 == 0:
            print('epoch {} || loss_total:{:.0f} '.format(i, loss.item()))

    return z


def run(path_type=True):
    """
    :param path_type: noise dir is True, normal dir is False
    :return:
    """
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device：", device)

    z = optimize_z()
    G, D = read_model()

    # dataset to iterator
    test_dataset = make_test_dataset(path_type)
    batch_iterator = iter(test_dataset)
    test_images = next(batch_iterator)
    test_images = test_images.to(device)

    fake_image = G(z)

    _, loss = anomaly_score(test_images, fake_image, D)
    loss = loss.cpu().detach().numpy()
    print("total loss：", np.round(loss, 0))

    fig = plt.figure(figsize=(20, 9))
    for i in range(0, 5):
        # real image
        plt.subplot(2, 5, i+1)
        plt.imshow(test_images[i][0].cpu().detach().numpy(), )

        # fake image
        plt.subplot(2, 5, 5+i+1)
        plt.imshow(fake_image[i][0].cpu().detach().numpy(), )
