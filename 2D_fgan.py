# from tensorflow import set_random_seed
import os
import numpy as np
import matplotlib.pyplot as plt
# from keras.models import Model
# from keras.layers import Input, Add
# from keras.layers.core import Dense
# from keras.optimizers import Adam
# from keras import losses
# import keras.backend as K
from custom_losses2 import com_conv
import torch
from torch import nn

from numpy.random import seed
seed(1)
torch.manual_seed(1)
torch.set_default_dtype(torch.float32)

###Training hyperparameters###
epoch = 30001
batch_size = 100


###Generator Hyperparameters###
alpha = 0.5
beta = 15

###Discriminator Hyperparameters###
gamma = 0.1


# gm = K.variable([1])

if not os.path.exists('./pictures'):
    os.makedirs('./pictures')


def animate(G, D, epoch, v_animate):
    plt.figure()
    xlist = np.linspace(0, 40, 40)
    ylist = np.linspace(0, 40, 40)
    X, Y = np.meshgrid(xlist, ylist)
    In = np.array(np.meshgrid(xlist, ylist)).T.reshape(-1, 2)

    D.to(torch.device('cpu'))
    In = torch.tensor(In).float()
    Out = D(In)
    In = In.detach().numpy()
    Out = Out.detach().numpy()

    Z = Out.reshape(40, 40).T
    c = ('#66B2FF', '#99CCFF', '#CCE5FF', '#FFCCCC', '#FF9999', '#FF6666')
    cp = plt.contourf(X, Y, Z, [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], colors=c)
    plt.colorbar(cp)

    rx, ry = data_D(G, 500, 'real')[0].T
    gx, gy = data_D(G, 200, 'gen')[0].T
    # plotting the sample data, generated data
    plt.scatter(rx, ry, color='red')
    plt.scatter(gx, gy, color='blue')
    plt.xlabel('x-axis')
    plt.xlim(0, 40)
    plt.ylabel('y-axis')
    plt.ylim(0, 40)
    plt.title('Epoch'+str(epoch))
    plt.savefig('./pictures/'+str(int(epoch/v_animate))+'.png', dpi=500)
    plt.close()


def real_data(n):
    return np.random.normal((20, 20), 3, [n, 2])


def noise_data(n):
    return np.random.normal(0, 8, [n, 2])

# Prepare training dataset for Discriminator


def data_D(G, n_samples, mode):
    if mode == 'real':
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

        def forward(self, inp):
            # print(inp.size())
            # x = inp.view(-1, 2)
            x = self.fc1(inp)
            x = self.activ(x)
            x = self.fc2(x)
            x = self.activ(x)
            x = self.fc3(x)

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


def pretrain(G, D, dopt, epoch=20, n_samples=batch_size):  # pretrain D
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for epoch in range(epoch):
        D.train()
        dopt.zero_grad()

        loss_temp = []
        set_trainability(D, True)

        x, y = data_D(G, n_samples, 'real')

        D = D.to(device)
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        output = D(x)
        loss1 = D_loss(y, output)

        loss_temp.append(loss1)

        set_trainability(D, True)
        x, y = data_D(G, n_samples, 'gen')

        D = D.to(device)
        x = torch.tensor(x).float().to(device)
        y = torch.tensor(y).float().to(device)
        output = D(x)
        loss2 = D_loss(y, output)

        loss_temp.append(loss2)

        loss = (loss1+loss2)/2
        loss.backward()
        dopt.step()

        print('Pretrain Epoch {} Dis Loss {}'.format(
            epoch, sum(loss_temp)/len(loss_temp)))


def train(GAN, G, D, dopt, gopt, epochs=epoch, n_samples=batch_size, v_freq=100, v_animate=1000):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    d_loss = []
    g_loss = []
#    data_show = sample_noise(n_samples=n_samples)[0]
    for epoch in range(epochs):
        try:
            D.train()
            dopt.zero_grad()

            loss_temp = []
            set_trainability(D, True)
            x, y = data_D(G, n_samples, 'real')
            D = D.to(device)
            x = torch.tensor(x).float().to(device)
            y = torch.tensor(y).float().to(device)
            output = D(x)
            loss1 = D_loss(y, output)

            loss_temp.append(loss1)

            set_trainability(D, True)
            x, y = data_D(G, n_samples, 'gen')
            D = D.to(device)
            x = torch.tensor(x).float().to(device)
            y = torch.tensor(y).float().to(device)
            output = D(x)
            loss2 = D_loss(y, output)

            loss_temp.append(loss2)

            loss = (loss1+loss2)/2
            loss.backward()
            dopt.step()

            d_loss.append(sum(loss_temp)/len(loss_temp))

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

            output = GAN(D, G, x)
            G_out = GAN.get_G_out()
            gan_criterion = com_conv(G_out, beta, 2)
            gan_loss = gan_criterion(y, output)

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

    torch.save(G.state_dict(), 'Circle_G.pth')
    torch.save(D.state_dict(), 'Circle_D.pth')

    return d_loss, g_loss


G = get_generative()
D, dopt = get_discriminative()
GAN, gopt = make_gan(G, D)
pretrain(G, D, dopt)
d_loss, g_loss = train(GAN, G, D, dopt, gopt)
