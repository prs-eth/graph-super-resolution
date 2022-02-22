import torch.nn.functional as f


def mse_loss_func(pred, gt, mask):
    return f.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    return f.l1_loss(pred[mask == 1.], gt[mask == 1.])
