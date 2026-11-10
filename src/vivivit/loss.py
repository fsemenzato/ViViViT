import torch
from torch.nn import functional as F


def reconstruction_loss(prediction, target, norm='L1'):
    loss = {'L1': lambda a, b: torch.abs(a-b).mean(), 'L2': F.mse_loss}[norm]
    pixel = F.l1_loss(prediction, target) if norm == 'L1' else F.mse_loss(prediction, target)
    dx = prediction[:, :, :, 1:, :]-prediction[:, :, :, :-1, :]
    dy = prediction[:, :, :, :, 1:]-prediction[:, :, :, :, :-1]
    tx = target[:, :, :, 1:, :]-target[:, :, :, :-1, :]
    ty = target[:, :, :, :, 1:]-target[:, :, :, :, :-1]
    dt = prediction[:, :, 1:, :, :]-prediction[:, :, :-1, :, :]
    tt = target[:, :, 1:, :, :]-target[:, :, :-1, :, :]
    return pixel+(loss(dx, tx)+loss(dy, ty)+loss(dt, tt))/3.
