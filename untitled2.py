# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 18:50:04 2023

@author: Timothy
"""

import tensorflow as tf
from torchvision import datasets
# from keras.datasets import mnist, cifar10
import numpy as np
import torch
from torch import nn
from math import floor, ceil


def Conv2DTranspose(X, W, padding="valid", strides=(1, 1)):
    # Define output shape before padding
    row_num = (X.shape[0] - 1) * strides[0] + W.shape[0]
    col_num = (X.shape[1] - 1) * strides[1] + W.shape[1]
    output = np.zeros([row_num, col_num])
    # Calculate the output
    for i in range(0, X.shape[0]):
        i_prime = i * strides[0]  # Index in output
        for j in range(0, X.shape[1]):
            j_prime = j * strides[1]
            # Insert values
            for k_row in range(W.shape[0]):
                for k_col in range(W.shape[1]):
                    output[i_prime+k_row, j_prime+k_col] += W[k_row, k_col] * X[i, j]
    # Define length of padding
    if padding == "same":
        # returns the output with the shape of (input shape)*(stride)
        p_left = floor((W.shape[0] - strides[0])/2)
        p_right = W.shape[0] - strides[0] - p_left
        p_top = floor((W.shape[1] - strides[1])/2)
        p_bottom = W.shape[1] - strides[1] - p_left
    elif padding == "valid":
        # returns the output without any padding
        p_left = 0
        p_right = 0
        p_top = 0
        p_bottom = 0
    # Add padding
    output_padded = output[p_left:output.shape[0]-p_right, p_top:output.shape[0]-p_bottom]
    return(np.array(output_padded))


class mydataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.randn(256)

    def __len__(self):
        return 16


class mnistnoisedataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.randn(200)

    def __len__(self):
        return 16


class imdataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.randn(3, 32, 32)

    def __len__(self):
        return 16


class immnistdataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.randn(1, 28, 28)

    def __len__(self):
        return 16


mnist_ds = mnistnoisedataset()
mnist_dl = torch.utils.data.DataLoader(mnist_ds, batch_size=8)

mnist_imds = immnistdataset()
mnist_imdl = torch.utils.data.DataLoader(mnist_imds, batch_size=8)

imds = imdataset()
imdl = torch.utils.data.DataLoader(imds, batch_size=8)

ds = mydataset()
dl = torch.utils.data.DataLoader(ds, batch_size=8)

# # a = a.double()
# # a = torch.nn.functional.pad(a, (1, 1, 1, 1))
# z = torch.nn.ConvTranspose2d(256, 128, kernel_size=(
#     5, 5), stride=2, padding=2, output_padding=1)(a)
# print(z.detach().numpy()[0, 0])
# w = torch.tensor(np.random.rand(5, 5))
# y = Conv2DTranspose(x, w, 'valid', (2, 2))

# x = torch.randn(256)


class Generator(nn.Module):
    '''
    Building Generator
    '''

    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(200, 1024)
        torch.nn.init.normal_(self.fc1.weight, std=0.02)
        self.activ1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 128*7*7)
        torch.nn.init.normal_(self.fc2.weight, std=0.02)
        self.activ2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128*7*7)

        self.conv1 = nn.ConvTranspose2d(
            128, 64, (4, 4), stride=2, padding=(1, 1))
        torch.nn.init.normal_(self.conv1.weight, std=0.02)
        self.activ3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)

        self.conv2 = nn.ConvTranspose2d(
            64, 1, (4, 4), stride=2, padding=(1, 1))
        torch.nn.init.normal_(self.conv2.weight, std=0.02)
        self.activ4 = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = x.view(-1, 1024, 1)
        x = self.bn1(x)
        x = x.view(-1, 1024)

        x = self.fc2(x)
        x = self.activ2(x)
        x = x.view(-1, 128*7*7, 1)
        x = self.bn2(x)
        x = x.view(-1, 128, 7, 7)

        x = self.conv1(x)
        x = self.activ3(x)
        x = self.bn3(x)

        x = self.conv2(x)
        x = self.activ4(x)

        return x


class Discriminator(nn.Module):
    '''
    Builiding Discriminator
    '''

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, (4, 4), stride=2, padding=(1, 1))
        torch.nn.init.normal_(self.conv1.weight, std=0.02)
        self.activ1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(64, 64, (4, 4), stride=2, padding=(1, 1))
        torch.nn.init.normal_(self.conv2.weight, std=0.02)
        self.activ2 = nn.LeakyReLU(0.1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*7*7, 1024)
        torch.nn.init.normal_(self.fc1.weight, std=0.02)
        self.activ3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(1024, 1)
        torch.nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.activ2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activ3(x)
        x = self.fc2(x)
        return x

# x = torch.randn(4, 32, 16, 16)
# x = nn.Conv2d(32, 64, (5, 5), stride=2, padding=(2, 2))(x)
# x = x.detach().numpy()

# input_shape = (4, 32, 32, 3)
# x = tf.random.normal(input_shape)
# y = tf.keras.layers.Conv2D(32, (5, 5), strides=2,
#                            kernel_initializer='glorot_normal', padding='same')(x)
# print(y.shape)

# x = torch.randn(3, 64, 14, 14)
# x = nn.Conv2d(64, 64, (4, 4), stride=2, padding=(1, 1))(x)

# g = G()
# par = []
# for p in g.parameters():
#     par.append(p)

# x = par[0]


x = torch.randn(2)
print(type(x.data.numpy()[0]))
x = x.float()
print(type(x.data.numpy()[0]))

# b = torch.arange(2 * 3 * 4).view(2, 3, 4)
# print(torch.sum(b, (2, 1)))
# print(torch.sum(b, (1, 2)))

# g = Generator()
# g = Discriminator()
# for x in mnist_imdl:
#     x = g(x)
# print(x)

# for m in g.modules():
#     print(m)

# # print(x.detach().numpy()[0, 0])
# a = []
# c = g.named_parameters()


def get_wd_params(model: nn.Module):
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():
        # print('checking {}'.format(name))
        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue
        # 'norm' not in name and 'bn' not in name and 'BatchN' not in name:
        if 'weight' in name and 'conv' in name:
            decay.append(param)
            # print(name, 'dec')
        else:
            no_decay.append(param)
            # print(name, 'no')
    return decay, no_decay


def get_wd_params2(model: nn.Module):
    # Parameters must have a defined order.
    # No sets or dictionary iterations.
    # See https://pytorch.org/docs/stable/optim.html#base-class
    # Parameters for weight decay.
    all_params = tuple(model.parameters())
    wd_params = list()
    for m in model.modules():
        if isinstance(
                m,
                (
                    # nn.Linear,
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
        ):
            wd_params.append(m.weight)
    # Only weights of specific layers should undergo weight decay.
    no_wd_params = [p for p in all_params if p not in wd_params]
    assert len(wd_params) + len(no_wd_params) == len(all_params), "Sanity check failed."
    return wd_params, no_wd_params


# decay, nodecay = get_wd_params(g)

# optim_kwargs = {'lr': 0.001, 'betas': (0.5, 0.999)}

# gopt = torch.optim.AdamW([{'params': nodecay, 'weight_decay': 0}, {
#                          'params': decay, 'weight_decay': 0.0001}], **optim_kwargs)

# x = gopt.param_groups

# x = nn.ConvTranspose2d(
#     1, 2, (2, 2), stride=2, padding=(1, 1))
# # torch.nn.init.normal_(x.weight, std=0.02)

# # x = nn.Linear(2, 2)
# for a in x.parameters():
#     print(a)

# torch.nn.init.normal_(x.weight, std=0.0001)


# def weight_init(x):
#     nn.init.normal_(x, std=0.0)
#     return x


# for a in x.parameters():
#     print(a)

# for a in x.parameters():
#     a.apply(weight_init)


# x = torch.normal(torch.zeros(16), torch.tensor(0.02).expand(16)).detach().numpy()

# for b in g.modules():
#     print(b)
# print(a)
