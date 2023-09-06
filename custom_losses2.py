import torch

# Average distance from the Center of Mass


def com_conv(G_out, beta, power):
    def dispersion_loss(y_true, y_pred):
        loss_b = torch.nn.BCEWithLogitsLoss()(
            y_pred.float(), y_true.float().view(-1, 1))

        center = torch.mean(G_out, dim=0, keepdims=True)
        distance_xy = torch.pow(torch.abs(torch.subtract(G_out, center)), power)
        distance = torch.sum(distance_xy, (1))
        avg_distance = torch.mean(torch.pow(distance, 1/power))
        loss_d = torch.reciprocal(avg_distance)

        loss = loss_b + beta*loss_d
        return loss
    return dispersion_loss
