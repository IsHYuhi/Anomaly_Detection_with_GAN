from collections import OrderedDict
from models.Generator import Generator
from models.Discriminator import Discriminator
from utils.data_set import make_datapath_list, GAN_Img_Dataset, ImageTransform
import matplotlib.pyplot as plt
import matplotlib
import torch
import os

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


batch_size = 8
z_dim = 20
fixed_z = torch.randn(batch_size, z_dim)
fixed_z = fixed_z.view(fixed_z.size(0), fixed_z.size(1), 1, 1)

fake_images = G(fixed_z.to(device))

train_img_list = make_datapath_list(num=1000)
mean = (0.5,)
std = (0.5,)
train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

batch_iterator = iter(train_dataloader)

# fetch first element
images = next(batch_iterator)


fig = plt.figure(figsize=(15, 6))
for i in range(0,5):
    #train is upside
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i][0].cpu().detach().numpy(), 'gray')

    # generated is bottom
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(fake_images[i][0].cpu().detach().numpy(), 'gray')
plt.show()
