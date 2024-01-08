import torch
from torch import nn
import numpy as np
# import gc

# Average distance from the Center of Mass


def outline_loss(G_out, real, omega, power=2):
    '''
    dist_mat = (batch_size, batch_size) [each row represents one G_out instance distances']
    '''
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    relu = nn.ReLU()
    # print('size of G_out:', G_out.size(), '\nsize of real:', real.size())

    # print('dist_mat', dist_mat)
    # print('max,min,std', torch.max(dist_mat), torch.min(dist_mat), torch.std(dist_mat))
    # if torch.count_nonzero(dist_mat) != 0:
    # print(torch.sum(dist_mat)/torch.count_nonzero(dist_mat))

    dist_mat = torch.cdist(G_out, real)

    dist_mat = omega - dist_mat
    dist_mat = relu(dist_mat)
    nonzero = torch.count_nonzero(dist_mat)

    if nonzero == 0:
        return 0

    result = torch.sum(dist_mat, dim=[0, 1]) / nonzero

    return result

# test outline loss
# a = torch.tensor([[0, 0], [1, 1]])
# b = torch.tensor([[0.001, 0.011], [0.099, 2]])
# c = outline_loss(a, b, 2e-4)
# print(c)


def com_conv(G_out, beta, power, bm):
    def dispersion_loss(y_true, y_pred):
        loss_b = torch.nn.BCEWithLogitsLoss()(
            y_pred.float(), y_true.float().view(-1, 1))

        center = torch.mean(G_out, dim=0, keepdims=True)
        distance_xy = torch.pow(torch.abs(torch.subtract(G_out, center)), power)
        # print('dist xy size bef', distance_xy.size())
        distance = torch.sum(distance_xy, 1)
        # print('dist xy size aft', distance_xy.size())
        avg_distance = torch.mean(torch.pow(distance, 1/power))
        loss_d = torch.reciprocal(avg_distance)

        loss = bm*loss_b + beta*loss_d
        return loss
    return dispersion_loss


def gen_asymm(size, factor=12):
    '''
    size is an iterable of len 2
    '''
    x_trans = (40-factor)/2
    y_trans = x_trans
    rand_arr = np.random.random(list(size))
    rand_out = np.empty(size)
    # import pdb
    # pdb.set_trace()
    for i in range(size[0]):
        if rand_arr[i][0] >= 0 and rand_arr[i][0] < 0.2:
            rand_out[i][0] = np.random.uniform(low=0.0, high=0.1)
            rand_out[i][1] = np.random.uniform(low=0.0, high=1)
            continue
        if rand_arr[i][0] >= 0.2 and rand_arr[i][0] < 0.4:
            rand_out[i][0] = np.random.uniform(low=0.1, high=0.2)
            rand_out[i][1] = np.random.uniform(low=0.0, high=1.0)
            continue
        if rand_arr[i][0] >= 0.4 and rand_arr[i][0] < 0.6:
            rand_out[i][0] = np.random.uniform(low=0.2, high=0.3)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.2)
            continue
        if rand_arr[i][0] >= 0.6 and rand_arr[i][0] < 0.8:
            rand_out[i][0] = np.random.uniform(low=0.3, high=0.4)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.1)
            continue
        if rand_arr[i][0] >= 0.8 and rand_arr[i][0] < 0.84:
            rand_out[i][0] = np.random.uniform(low=0.4, high=0.5)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.2)
            continue
        if rand_arr[i][0] >= 0.84 and rand_arr[i][0] < 0.88:
            rand_out[i][0] = np.random.uniform(low=0.5, high=0.6)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.2)
            continue
        if rand_arr[i][0] >= 0.88 and rand_arr[i][0] < 0.92:
            rand_out[i][0] = np.random.uniform(low=0.6, high=0.7)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.2)
            continue
        if rand_arr[i][0] >= 0.92 and rand_arr[i][0] < 0.96:
            rand_out[i][0] = np.random.uniform(low=0.7, high=0.8)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.2)
            continue
        if rand_arr[i][0] >= 0.96 and rand_arr[i][0] < 0.98:
            rand_out[i][0] = np.random.uniform(low=0.8, high=0.9)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.2)
            continue
        if rand_arr[i][0] >= 0.98 and rand_arr[i][0] <= 1.0:
            rand_out[i][0] = np.random.uniform(low=0.9, high=1.0)
            rand_out[i][1] = np.random.uniform(low=0.0, high=0.3)
            continue
        else:
            raise Exception('Invalid Value')

    return rand_out*factor + np.array([x_trans, y_trans])

# test gen_asymm
# a = gen_asymm([10, 2])
