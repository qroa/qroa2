import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, HuberLoss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, values, score):
        # Sum the result
        score = torch.log(score)
        loss = (values - score) ** 2

        return loss.mean()