import torch
import torch.nn as nn
import torch.nn.functional as F


class ONNXAttentiveStatsPool(nn.Module):
    def __init__(self, channels, attention_channels=128):
        super().__init__()
        self.linear1 = nn.Conv1d(channels, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)

        # attention weights
        h = torch.tanh(self.linear1(x))
        w = self.linear2(h)                # (B, C, T)
        w = F.softmax(w, dim=2)            # attention across time

        # weighted mean
        mean = torch.sum(w * x, dim=2)

        # weighted std
        var = torch.sum(w * (x - mean.unsqueeze(2)) ** 2, dim=2)
        std = torch.sqrt(var + 1e-6)

        # concat
        out = torch.cat([mean, std], dim=1)
        return out
