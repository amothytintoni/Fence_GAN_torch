# import tensorflow as tf
# from keras import losses
import torch


# Average distance from the Center of Mass
def com_conv(G_out, beta, power):
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

        loss = loss_b + beta*loss_d
        return loss
    return dispersion_loss
