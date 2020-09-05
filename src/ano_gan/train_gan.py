import numpy as np
import time
import glob

import torch
import torch.utils.data as data
import torch.nn as nn

from src.ano_gan.gan import GAN_Img_Dataset, ImageTransform, Generator, Discriminator
from src.utils import read_config


def read_data():
    config_ini = read_config.read_config()
    train_path = config_ini.get('PATH', 'train_images')
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
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_model(generator, discriminator, dataloader):
    """
    """
    # Parameters -------------------------------------------------------
    config_ini = read_config.read_config()
    g_lr = config_ini.getfloat('TRAIN', 'g_learning_rate')
    d_lr = config_ini.getfloat('TRAIN', 'g_learning_rate')
    beta1_g = config_ini.getfloat('TRAIN', 'g_beta_1')
    beta2_g = config_ini.getfloat('TRAIN', 'g_beta_2')
    beta1_d = config_ini.getfloat('TRAIN', 'd_beta_1')
    beta2_d = config_ini.getfloat('TRAIN', 'd_beta_2')
    z_dim = config_ini.getint('GENERATOR', 'z_dim')
    num_epochs = config_ini.getint('TRAIN', 'num_epoch')
    # ------------------------------------------------------------------

    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device：", device)

    # Define optimizer and loss function
    g_optimizer = torch.optim.Adam(generator.parameters(), g_lr, [beta1_g, beta2_g])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), d_lr, [beta1_d, beta2_d])
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # To GPU and train mode
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    # Accelerator
    torch.backends.cudnn.benchmark = True

    # num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    iteration = 1
    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('-------------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-------------')
        print('（train）')

        for imges in dataloader:

            # Discriminator---------------------------------------------------------------------------------------------
            if imges.size()[0] == 1:  # If mini_batch is 1 will cause error. For avoid it.
                continue

            imges = imges.to(device)
            mini_batch_size = imges.size()[0]

            # make label
            real_l = np.random.randint(7, 12, (mini_batch_size)) / 10
            fake_l = np.random.randint(0, 2, (mini_batch_size)) / 10
            label_real = torch.from_numpy(real_l).to(device)
            label_fake = torch.from_numpy(fake_l).to(device)

            # judge real image
            d_out_real, _ = discriminator(imges)

            # judge fake image
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = generator(input_z)
            d_out_fake, _ = discriminator(fake_images)

            # loss
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Generator-------------------------------------------------------------------------------------------------

            # judge fake image
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = generator(input_z)
            d_out_fake, _ = discriminator(fake_images)

            # loss
            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # save loss
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        # print loss
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss / batch_size, epoch_g_loss / batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    print("Iteration:", iteration)

    return generator, discriminator


def run():
    """
    """
    # parameters----------------------------------------------------
    config_ini = read_config.read_config()
    z_dim = config_ini.getint('GENERATOR', 'z_dim')
    g_img_size = config_ini.getint('GENERATOR', 'image_size')
    d_img_size = config_ini.getint('DISCRIMINATOR', 'image_size')
    g_path = config_ini.get('PATH', 'g_save')
    d_path = config_ini.get('PATH', 'd_save')
    # --------------------------------------------------------------
    # set up data
    image_list = read_data()
    data_loader = make_dataset(image_list)

    # set up model
    G = Generator(z_dim, g_img_size)
    D = Discriminator(d_img_size)
    G.apply(weights_init)
    D.apply(weights_init)

    G_learned, D_learned = train_model(G, D, data_loader)

    # save model
    torch.save(D_learned, d_path)
    torch.save(G_learned, g_path)
