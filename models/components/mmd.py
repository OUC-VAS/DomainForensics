import torch
import torch.nn as nn


class MMDLoss(nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self, f_x, f_t):
        delta = f_x - f_t
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss