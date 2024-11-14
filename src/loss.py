import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, w1=1.0, w2=1.0):
        super(WeightedBCELoss, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, y_hat, y):
        # Calcul de la Binary Cross-Entropy pondérée
        loss = - (self.w1 * y * torch.log(y_hat + 1e-8) + self.w2 * (1 - y) * torch.log(1 - y_hat + 1e-8))
        return loss.mean()