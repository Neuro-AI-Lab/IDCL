import torch
import torch.nn as nn

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.register_buffer('weight', weight) if weight is not None else None
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, mask):
        device = pred.device 

        mask = mask.to(device)
        target = target.to(device)
        mask_ = mask.view(-1, 1)

        loss = self.loss(pred * mask_, target) / torch.sum(mask)

        return loss