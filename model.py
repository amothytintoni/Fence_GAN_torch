from custom_losses import *
import torch
from torch import nn
# import tensorflow as tf

# import keras.backend as K
# from keras import losses
# from keras.models import Model
# from keras.layers import Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU
# from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
# from keras.regularizers import l2
# from keras.optimizers import Adam

# gamma = K.variable([1])


def get_wd_params(model: nn.Module):
    decay = list()
    no_decay = list()
    for name, param in model.named_parameters():

        if hasattr(param, 'requires_grad') and not param.requires_grad:
            continue

        if 'weight' in name and 'conv' in name:
            decay.append(param)
            # print(name, 'dec')
        else:
            no_decay.append(param)
            # print(name, 'no')

    return decay, no_decay


def load_model(args):
    if args.dataset == 'mnist':
        return get_mnist_model(args)
    if args.dataset == 'cifar10':
        return get_cifar10_model(args)


# alternate to freeze D network while training only G in (G+D) combination
def set_trainability(model, trainable=False):
    model.requires_grad_ = trainable
    for param in model.parameters():
        param.requires_grad = trainable


def D_loss(y_true, y_pred, logits=True):
    if logits:
        loss_gen = nn.BCEWithLogitsLoss()(y_pred.float(), y_true.float().view(-1, 1))
    else:
        loss_gen = nn.BCELoss()(y_pred.float(), y_true.float().view(-1, 1))

    # loss = gamma * loss_gen
    # gamma multiplication is moved to the calling function as pytorch does not have keras backend variable saving equivalent

    return loss_gen


def get_cifar10_model(args):
    '''
    Return: G, D, GAN model, dopt, gopt (optims of D and GAN)
    '''

    class Generator_Cifar(nn.Module):
        '''
        Generator Class
        '''
        # padding=same formula: padding = ((stride-1)*input_image_size - stride + filter_size)/2
        # transpose convolution desired_output_size = input_size*stride
        # transpose convolution original_output_size = (inp_size-1)*stride + kernel_size
        # hence transpose padding = kernel_size - stride [for each dimension]

        def __init__(self):
            super(Generator_Cifar, self).__init__()
            self.fc1 = nn.Linear(256, 256*2*2)
            self.bn1 = nn.BatchNorm2d(256)
            self.activ1 = nn.LeakyReLU(negative_slope=0.2)
            self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=(
                5, 5), stride=2, padding=2, output_padding=1)

            self.bn2 = nn.BatchNorm2d(128)
            self.activ2 = nn.LeakyReLU(negative_slope=0.2)
            self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=(
                5, 5), stride=2, padding=2, output_padding=1)

            self.bn3 = nn.BatchNorm2d(64)
            self.activ3 = nn.LeakyReLU(negative_slope=0.2)
            self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=(
                5, 5), stride=2, padding=2, output_padding=1)

            self.bn4 = nn.BatchNorm2d(32)
            self.activ4 = nn.LeakyReLU(negative_slope=0.2)
            self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=(
                5, 5), stride=2, padding=2, output_padding=1)

            self.activ5 = nn.Tanh()

        def forward(self, x):
            x = self.fc1(x)
            x = x.view(-1, 256, 2, 2)
            x = self.bn1(x)
            x = self.activ1(x)
            x = self.conv1(x)  # size becomes 128*4*4

            x = self.bn2(x)
            x = self.activ2(x)
            x = self.conv2(x)

            x = self.bn3(x)
            x = self.activ3(x)
            x = self.conv3(x)

            x = self.bn4(x)
            x = self.activ4(x)
            x = self.conv4(x)

            x = self.activ5(x)

            return x

    class Discriminator_Cifar(nn.Module):
        '''
        Build Discriminator
        '''

        def __init__(self):
            super(Discriminator_Cifar, self).__init__()
            # incl regularizer in optim later
            self.conv1 = nn.Conv2d(3, 32, (5, 5), stride=2, padding=(2, 2))
            self.bn1 = nn.BatchNorm2d(32)
            self.activ1 = nn.LeakyReLU(negative_slope=0.2)

            self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=2, padding=(2, 2))
            self.bn2 = nn.BatchNorm2d(64)
            self.activ2 = nn.LeakyReLU(negative_slope=0.2)

            self.conv3 = nn.Conv2d(64, 128, (5, 5), stride=2, padding=(2, 2))
            self.bn3 = nn.BatchNorm2d(128)
            self.activ3 = nn.LeakyReLU(negative_slope=0.2)

            self.conv4 = nn.Conv2d(128, 256, (5, 5), stride=2, padding=(2, 2))
            self.bn4 = nn.BatchNorm2d(256)
            self.activ4 = nn.LeakyReLU(negative_slope=0.2)

            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(0.2)
            self.linear1 = nn.Linear(256*2*2, 1)
            # dont need sigmoid as we can utilize BCEWithLogitsLoss

        def forward(self, x):
            x = self.conv1(x)  # incl regularizer in optim later
            x = self.bn1(x)
            x = self.activ1(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.activ2(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.activ3(x)

            x = self.conv4(x)
            x = self.bn4(x)
            x = self.activ4(x)

            x = self.flatten(x)
            x = self.dropout(x)
            x = self.linear1(x)

            return x

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

    D = Discriminator_Cifar()
    G = Generator_Cifar()

    # only decay the conv params
    decay, nodecay = get_wd_params(D)

    optim_kwargs = {'lr': args.d_lr, 'betas': (0.5, 0.999)}
    dopt = torch.optim.AdamW([{'params': nodecay, 'weight_decay': 0}, {
                             'params': decay, 'weight_decay': 1e-5 + args.d_l2}], **optim_kwargs)
    # AdamW is more adapted to weight_decay than Adam
    # decay rate is added by l2 reg because in the original implementation, the Conv params are regularized with l2 and the optimizer is regularized with decay. PyTorch does not have innate l2 regularizer in the Conv module, so the regularization is added in the decay which is roughly an L2 regularization.

    set_trainability(D, False)
    GAN = GAN_model()

    gopt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                   G.parameters()), lr=args.g_lr, betas=(0.5, 0.999))

    return G, D, GAN, dopt, gopt


def get_mnist_model(args):
    '''
    Return: G, D, GAN
    '''

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
            # dont need sigmoid as we can utilize BCEWithLogitsLoss

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

    D = Discriminator()
    G = Generator()
    dopt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                   D.parameters()), lr=args.d_lr, betas=(0.5, 0.999))
    set_trainability(D, False)

    GAN = GAN_model()
    gopt = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                   G.parameters()), lr=args.g_lr, betas=(0.5, 0.999))
    # GAN_loss = com_conv(G_out, args.beta, 2)

    return G, D, GAN, dopt, gopt
