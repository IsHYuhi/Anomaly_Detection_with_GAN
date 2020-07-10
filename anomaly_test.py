from utils.data_set import GAN_Img_Dataset, ImageTransform
from collections import OrderedDict
from models.Generator import Generator
from models.Discriminator import Discriminator
from utils.Anomaly_score import Anomaly_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
import os

def make_datapath_list(root_path="./data/img", num=200):
    """
    make filepath list for train and validation image and annotation.
    """

    #numberごとに均一枚数を取得
    train_img_list = []
    for img_idx in range(num):
        img_path = root_path + "/img_" + str(7) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

        img_path = root_path + "/img_" + str(8) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

        img_path = root_path + "/img_" + str(2) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

    return train_img_list

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

#torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


device = "cuda" if torch.cuda.is_available() else "cpu"
G = Generator(z_dim=20, image_size=64)
D = Discriminator(z_dim=20, image_size=64)

'''-------load weights-------'''
G_load_weights = torch.load('./checkpoints/G_AnoGAN_300.pth')
G.load_state_dict(fix_model_state_dict(G_load_weights))

D_load_weights = torch.load('./checkpoints/D_AnoGAN_300.pth')
D.load_state_dict(fix_model_state_dict(D_load_weights))

G.to(device)
D.to(device)

"""use GPU in parallel"""
if device == 'cuda':
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)
    print("parallel mode")


batch_size = 5

train_img_list = make_datapath_list(num=1000)
mean = (0.5,)
std = (0.5,)
train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

batch_iterator = iter(train_dataloader)

# fetch first element
images = next(batch_iterator)



x = images[0:5]
x = x.to(device)

z = torch.randn(5, 20).to(device)
z = z.view(z.size(0), z.size(1), 1, 1)

# set requires_grad to True to calculate the derivative of the variable z.
z.requires_grad = True
z_optimizer = torch.optim.Adam([z], lr=1e-3)

for epoch in range(5000+1):
    fake_img = G(z)
    loss, _, _ = Anomaly_score(x, fake_img, D, Lambda=0.1)# using D

    z_optimizer.zero_grad()
    loss.backward()# find the derivative of z in the direction of lowering loss.
    z_optimizer.step()# update

    if epoch % 1000 == 0:
        print('epoch {} || loss_total:{:.0f} '.format(epoch, loss.item()))


fake_img = G(z)

loss, loss_each, residual_loss_each = Anomaly_score(x, fake_img, D, Lambda=0.1)

loss_each = loss_each.cpu().detach().numpy()
print("total loss：", np.round(loss_each, 0))


fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
    # testdata
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i][0].cpu().detach().numpy(), 'gray')

    # generated
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(fake_img[i][0].cpu().detach().numpy(), 'gray')

plt.show()
