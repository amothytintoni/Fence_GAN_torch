# import tensorflow as tf
# from keras import losses
import torch
from torch import nn
import gc
from scipy.spatial.distance import cdist


def outline_loss(G_out, real, omega, power=2, verif=False):
    '''
    dist_mat = (batch_size, batch_size) [each row represents one G_out instance distances']
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    relu = nn.ReLU()
    # print('size of G_out:', G_out.size(), '\nsize of real:', real.size())

    if len(G_out.size()) == 4:  # (n_batch, n_channel, l, w) -> (n_batch, n_dim)
        G_out = torch.sum(G_out, dim=1)
        dims = G_out.size()
        G_out = G_out.view(dims[0], -1)
        real = torch.sum(real, dim=1)
        dims = real.size()
        real = real.view(dims[0], -1)
    elif len(G_out.size()) == 3:  # (n_batch, l, w) -> (n_batch, n_dim)
        dims = G_out.size()
        G_out = G_out.view(dims[0], -1)
        dims = real.size()
        real = real.view(dims[0], -1)

    # print('size of G_out:', G_out.size(), '\nsize of real:', real.size())

    dist_mat = torch.cdist(G_out, real)

    if verif:
        print('dist_mat', dist_mat)
        print('max,min,std', torch.max(dist_mat),
              torch.min(dist_mat), torch.std(dist_mat))
        if torch.count_nonzero(dist_mat) != 0:
            print(torch.sum(dist_mat)/torch.count_nonzero(dist_mat))

    dist_mat = omega - dist_mat
    dist_mat = relu(dist_mat)
    nonzero = torch.count_nonzero(dist_mat)

    if nonzero == 0:
        return 0

    result = torch.sum(dist_mat, dim=[0, 1]) / nonzero

    return result


# test outline loss
# a = torch.tensor([[0, 0, 0], [1, 1, 1]])
# b = torch.tensor([[0.001, 0.011, 0], [0.099, 2, 1]])
# c = outline_loss(a, b, 2e-4)
# print(c)

# Average distance from the Center of Mass


def com_conv(G_out, beta, power, bm):
    def dispersion_loss(y_true, y_pred):
        # print(torch.max(y_true), torch.min(y_true))
        # print(torch.max(y_pred), torch.min(y_pred))
        # print('size', y_true.size(), y_pred.size())

        loss_b = torch.nn.BCEWithLogitsLoss()(
            y_pred.float(), y_true.float().view(-1, 1))

        center = torch.mean(G_out, dim=0, keepdims=True)
        distance_xy = torch.pow(torch.abs(torch.subtract(G_out, center)), power)
        distance = torch.sum(distance_xy, (1, 2, 3))
        avg_distance = torch.mean(torch.pow(torch.abs(distance), 1/power))
        loss_d = torch.reciprocal(avg_distance)

        loss = bm*loss_b + beta*loss_d
        return loss
    return dispersion_loss
