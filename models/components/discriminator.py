import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Discriminator, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim

        self.discriminator = nn.Sequential(
            nn.Linear(self._input_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._out_dim),
        )

        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, feats):
        out = self.discriminator(feats)
        return out
