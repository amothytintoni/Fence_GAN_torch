

# from tensorflow import set_random_seed
from copy import deepcopy
import os
import numpy as np
import matplotlib.pyplot as plt
# from keras.models import Model
# from keras.layers import Input, Add
# from keras.layers.core import Dense
# from keras.optimizers import Adam
# from keras import losses
# import keras.backend as K
from util import com_conv, gen_asymm, outline_loss
import torch
from torch import nn
import argparse
import time

from numpy.random import seed
seed(4)
torch.manual_seed(4)
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser('2D synthetic Data Fence GAN')
parser.add_argument('--ol', '-ol', action='store_false', help='disable outline loss')
parser.add_argument('--asym', '-asym', action='store_false',
                    help='disable asymmetric data, revert back to original circular data')
parser.add_argument('--kappa', '-k', type=float, default=0.5,
                    help='outline loss weighting hyperparameter')
parser.add_argument('--folder_naming', '-fn', action='store_false',
                    help='disable automatically unique naming the result folder for each run')
parser.add_argument('--omega', '-o', type=float, default=1.2,
                    help='closeness hyperparameter for outline loss')
parser.add_argument('--plot_contour', '-c', action='store_false',
                    help='disable generation of contour+real data plots')
parser.add_argument('--contour_only', '-co', action='store_true',
                    help='generate contour only plots')
parser.add_argument('--alpha', '-a', type=float, default=0.5, help='alpha hyperparameter')
parser.add_argument('--beta', '-b', type=float, default=15, help='beta hyperparameter')
parser.add_argument('--gamma', '-g', type=float, default=0.1, help='gamma hyperparameter')
args = parser.add_argument('--bm', '-bm', type=float, default=1,
                           help='weighting hyperparameter for encirclement loss')
args = parser.parse_args()

###Training hyperparameters###
epoch = 40001
batch_size = 100
pretrain_epoch = 15
v_freq = 1000
freq_animate = 1000


###Generator Hyperparameters###
alpha = args.alpha
beta = args.beta
KAPPA = args.kappa  # outline loss weighting hyperparameter

###Discriminator Hyperparameters###
gamma = args.gamma

# gm = K.variable([1])

now = int(time.time())
if args.folder_naming and not os.path.exists(f'./pictures{now}'):
    os.makedirs(f'./pictures{now}')


def animate(G, D, epoch, v_animate):
    plt.figure()
    xlist = np.linspace(0, 40, 40)
    ylist = np.linspace(0, 40, 40)
    X, Y = np.meshgrid(xlist, ylist)
    In = np.array(np.meshgrid(xlist, ylist)).T.reshape(-1, 2)

    D = D.to(torch.device('cpu'))
    In = torch.tensor(In).float()
    Out = D(In)
    In = In.detach().numpy()
    Out = Out.detach().numpy()

    Z = Out.reshape(40, 40).T
    c = ('#66B2FF', '#99CCFF', '#CCE5FF', '#FFCCCC', '#FF9999', '#FF6666')
    cp = plt.contourf(X, Y, Z, [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], colors=c)
    plt.colorbar(cp)

    plt.xlabel('x-axis')
    plt.xlim(0, 40)
    plt.ylabel('y-axis')
    plt.ylim(0, 40)

    if args.contour_only:
        plt.title('Epoch'+str(epoch))
        plt.savefig(f'./pictures{now}/'+str(int(epoch/v_animate))+'_oc.png', dpi=500)

    rx, ry = data_D(G, 350, 'real')[0].T
    plt.scatter(rx, ry, color='red')

    if args.plot_contour:

        plt.title('Epoch'+str(epoch))
        plt.savefig(f'./pictures{now}/'+str(int(epoch/v_animate))+'_c.png', dpi=500)

    gx, gy = data_D(G, 230, 'gen')[0].T
    # plotting the sample data, generated data
    plt.scatter(gx, gy, color='blue')

    plt.title('Epoch'+str(epoch))
    plt.savefig(f'./pictures{now}/'+str(int(epoch/v_animate))+'.png', dpi=500)
    plt.close()


def real_data(n):
    return np.random.normal((20, 20), 3, [n, 2])


def noise_data(n):
    return np.random.normal(0, 8, [n, 2])

# Prepare training dataset for Discriminator


def data_D(G, n_samples, mode):
    if mode == 'real':
        if args.asym:
            x = gen_asymm([n_samples, 2])
        else:
            x = real_data(n_samples)

        y = np.ones(n_samples)
        return x, y

    elif mode == 'gen':
        noise = torch.tensor(noise_data(n_samples)).float()
        G = G.to(torch.device('cpu'))
        x = G(noise)
        y = np.zeros(n_samples)
        x = x.detach().numpy()
        return x, y

# Prepare training dataset for Generator


def data_G(batch_size):
    x = noise_data(batch_size)
    y = np.zeros(batch_size)
    y[:] = alpha
    return x, y

# Discriminator Loss function


def D_loss(y_true, y_pred):
    loss_gen = nn.BCEWithLogitsLoss()(y_pred.float(), y_true.float().view(-1, 1))
    loss = gamma*loss_gen
    return loss

# Generator model


def get_generative():
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 10)
            self.fc3 = nn.Linear(10, 2)
            self.activ = nn.ReLU()

        def forward(self, inp):
            # print(inp.size())
            # x = inp.view(-1, 2)
            x = self.fc1(inp)
            x = self.activ(x)
            x = self.fc2(x)
            x = self.activ(x)
            x = self.fc3(x)
            x = torch.add(inp, x)
            return x

    G = Generator()
    return G

# Discriminator model


def get_discriminative():
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.fc1 = nn.Linear(2, 15)
            self.fc2 = nn.Linear(15, 15)
            self.fc3 = nn.Linear(15, 1)
            self.activ = nn.ReLU()
            # self.sigm = nn.Sigmoid()

        def forward(self, inp):
            # print(inp.size())
            # x = inp.view(-1, 2)
            x = self.fc1(inp)
            x = self.activ(x)
            x = self.fc2(x)
            x = self.activ(x)
            x = self.fc3(x)
            # x = self.sigm(x)

            return x

    D = Discriminator()
    dopt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                   D.parameters()), lr=1e-3)

    return D, dopt


def set_trainability(model, trainable=False):
    model.requires_grad_ = trainable
    for param in model.parameters():
        param.requires_grad = trainable


def make_gan(G, D):  # making (G+D) framework
    class GAN_model(nn.Module):
        '''
        Building GAN
        '''

        def get_G_out(self):
            return self.G_out

        def forward(self, D, G, GAN_in):
            G_out = G(GAN_in)
            self.G_out = G_out
            GAN_out = D(G_out)
            return GAN_out

    set_trainability(D, False)
    GAN = GAN_model()
    gopt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                   G.parameters()), lr=1e-3)

    return GAN, gopt


def pretrain(G, D, dopt, epoch=pretrain_epoch, n_samples=batch_size):  # pretrain D
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # dopt2 = deepcopy(dopt)
    for epoch in range(epoch):
        D.train()
        dopt.zero_grad()
        # dopt2.zero_grad()

        loss_temp = []
        set_trainability(D, True)

        x, y = data_D(G, n_samples, 'real')

        D = D.to(device)
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        output = D(x)
        loss1 = D_loss(y, output)

        loss_temp.append(loss1)
        loss1.backward()
        dopt.step()
        dopt.zero_grad()

        set_trainability(D, True)
        x, y = data_D(G, n_samples, 'gen')

        D = D.to(device)
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        output = D(x)
        loss2 = D_loss(y, output)

        loss_temp.append(loss2)
        loss2.backward()
        dopt.step()

        loss = (loss1+loss2)/2
        # loss.backward()
        # dopt.step()

        print('Pretrain Epoch {} Dis Loss {}'.format(
            epoch, sum(loss_temp)/len(loss_temp)))


def train(GAN, G, D, dopt, gopt, epochs=epoch, n_samples=batch_size, v_freq=v_freq, v_animate=freq_animate):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    d_loss = []
    g_loss = []
    # dopt2 = deepcopy(dopt)
#    data_show = sample_noise(n_samples=n_samples)[0]
    for epoch in range(epochs):
        try:
            D.train()
            dopt.zero_grad()
            # dopt2.zero_grad()

            loss_temp = []
            set_trainability(D, True)
            x, y = data_D(G, n_samples, 'real')
            D = D.to(device)
            x = torch.tensor(x).float().to(device)
            y = torch.tensor(y).float().to(device)
            realdata = deepcopy(x)
            output = D(x)
            loss1 = D_loss(y, output)

            loss_temp.append(loss1)
            loss1.backward()
            dopt.step()
            dopt.zero_grad()

            set_trainability(D, True)
            x, y = data_D(G, n_samples, 'gen')
            D = D.to(device)
            x = torch.tensor(x).float().to(device)
            y = torch.tensor(y).float().to(device)
            output = D(x)
            loss2 = D_loss(y, output)

            loss_temp.append(loss2)
            loss2.backward()
            dopt.step()

            loss = (loss1+loss2)/2
            # loss.backward()
            # dopt.step()

            d_loss.append(loss)

            # Train Generator
            GAN.train()
            G.train()
            gopt.zero_grad()

            set_trainability(D, False)
            X, y = data_G(n_samples)
            X = torch.tensor(X).float().to(device)
            y = torch.tensor(y).float().to(device)
            GAN = GAN.to(device)
            G = G.to(device)

            output = GAN(D, G, X)
            G_out = GAN.get_G_out()
            gan_criterion = com_conv(G_out, beta, 2, args.bm)
            gan_loss = gan_criterion(y, output)

            if args.ol:
                ol_loss = outline_loss(G_out, realdata, args.omega)
                if (epoch + 1) % v_freq == 0:
                    print('GAN loss=', gan_loss, 'Outline Loss=', ol_loss)
                gan_loss += KAPPA * ol_loss

            g_loss.append(gan_loss)

            gan_loss.backward()
            gopt.step()

            G.eval()
            D.eval()
            GAN.eval()

            if (epoch + 1) % v_freq == 0:
                print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(
                    epoch + 1, g_loss[-1], d_loss[-1]))
            if epoch % v_animate == 0:
                animate(G, D, epoch, v_animate)
        except KeyboardInterrupt:
            break

    torch.save(G.state_dict(), f'Circle_G_{now}.pth')
    torch.save(D.state_dict(), f'Circle_D_{now}.pth')

    return d_loss, g_loss


G = get_generative()
D, dopt = get_discriminative()
GAN, gopt = make_gan(G, D)
pretrain(G, D, dopt)
d_loss, g_loss = train(GAN, G, D, dopt, gopt)
