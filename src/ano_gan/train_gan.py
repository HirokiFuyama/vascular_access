import numpy as np
import time

import torch
import torch.utils.data as data
import torch.nn as nn

import glob

from ano_gan.gan import GAN_Img_Dataset
from ano_gan.gan import ImageTransform

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

# def weights_init(m):
#     """
#     This function initialize Conv2d, ConvTranspose2d and BatchNorm2d.
#     :param model: generator or discriminator
#     """
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         # Conv2dとConvTranspose2dの初期化
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
#     elif classname.find('BatchNorm') != -1:
#         # BatchNorm2dの初期化
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


def train_model(generator, discriminator, dataloader, num_epochs):
    """
    :param generator:
    :param discriminator:
    :param dataloader:
    :param num_epochs:
    :return:
    """
    # Parameters -------------------------------------------------------
    config_ini = read_config.read_config()
    g_lr = config_ini.getfloat('TRAIN', 'g_learning_rate')
    d_lr = config_ini.getfloat('TRAIN', 'g_learning_rate')
    beta1_g = config_ini.getfloat('TRAIN', 'g_beta_1')
    beta2_g = config_ini.getfloat('TRAIN', 'g_beta_2')
    beta1_d = config_ini.getfloat('TRAIN', 'd_beta_1')
    beta2_d = config_ini.getfloat('TRAIN', 'd_beta_2')
    z_dim = config_ini.getfint('GENERATOR', 'z_dim')
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

            # Discriminator----------------------------------------------------------------------------
            if imges.size()[0] == 1:  # If mini_batch is 1 will cause error. For avoid it.
                continue

            imges = imges.to(device)

            # 正解ラベルと偽ラベルを作成
            # epochの最後のイテレーションはミニバッチの数が少なくなる
            mini_batch_size = imges.size()[0]

            real_l = np.random.randint(7, 12, (mini_batch_size)) / 10
            fake_l = np.random.randint(0, 2, (mini_batch_size)) / 10
            label_real = torch.from_numpy(real_l).to(device)
            label_fake = torch.from_numpy(fake_l).to(device)

            # 真の画像を判定
            d_out_real, _ = discriminator(imges)

            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = generator(input_z)
            d_out_fake, _ = discriminator(fake_images)

            # 誤差を計算
            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # --------------------
            # 2. Generatorの学習
            # --------------------
            # 偽の画像を生成して判定
            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = generator(input_z)
            d_out_fake, _ = discriminator(fake_images)

            # 誤差を計算
            g_loss = criterion(d_out_fake.view(-1), label_real)

            # バックプロパゲーション
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------------------
            # 3. 記録
            # --------------------
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_D_Loss:{:.4f} ||Epoch_G_Loss:{:.4f}'.format(
            epoch, epoch_d_loss / batch_size, epoch_g_loss / batch_size))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

    print("総イテレーション回数:", iteration)

    return generator, discriminator

# test -----------------------------------------------------------------------------------------------------------------
# def check_dataloader():
#     data = read_data()
#     data_loader = make_dataset(data)
#     batch_iterator = iter(data_loader)
#     imges = next(batch_iterator)
#     print(imges.size())
#
# check_dataloader()