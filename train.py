from collections import OrderedDict
from models.Generator import Generator
from models.Discriminator import Discriminator
from utils.data_set import make_datapath_list, GAN_Img_Dataset, ImageTransform
from torchvision import models
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch
import os

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #initialize Conv2d and ConvTranspose2d
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        #initialize BatchNorm2d
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def plot_log(data, save_model_name='model'):
    plt.cla()
    plt.plot(data['G'], label='G_loss ')
    plt.plot(data['D'], label='D_loss ')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs/'+save_model_name+'.png')

def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

def train_model(G, D, dataloader, num_epochs, save_model_name='model'):

    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G.to(device)
    D.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        print("parallel mode")

    print("device:{}".format(device))
    g_lr, d_lr = 0.0001, 0.0004
    beta1, beta2 = 0.0, 0.9
    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    z_dim = 20
    mini_batch_size = 256

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    g_losses = []
    d_losses = []
    losses = {'G':g_losses, 'D':d_losses}

    for epoch in range(num_epochs+1):

        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('-----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('(train)')

        for images in tqdm(dataloader):

            # Train Discriminator
            # if size of minibatch is 1, an error would be occured.
            if images.size()[0] == 1:
                continue

            images = images.to(device)

            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            d_out_real, _ = D(images)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()

            # Train Generator

            input_z = torch.randn(mini_batch_size, z_dim).to(device)
            input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        losses['G'].append(epoch_g_loss/batch_size)
        losses['D'].append(epoch_d_loss/batch_size)

        plot_log(losses, save_model_name)

        if(epoch%10 == 0):
            torch.save(G.state_dict(), 'checkpoints/G_'+save_model_name+'_'+str(epoch)+'.pth')
            torch.save(D.state_dict(), 'checkpoints/D_'+save_model_name+'_'+str(epoch)+'.pth')

    return G, D

def main():
    G = Generator(z_dim=20, image_size=64)
    D = Discriminator(z_dim=20, image_size=64)
    G.apply(weights_init)
    D.apply(weights_init)

    train_img_list=make_datapath_list(num=1000)
    mean = (0.5,)
    std = (0.5,)
    train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))

    batch_size = 256
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 100
    G_update, D_update = train_model(G, D, dataloader=train_dataloader, num_epochs=num_epochs, save_model_name='AnoGAN')


if __name__ == "__main__":
    main()
