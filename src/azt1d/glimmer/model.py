"""
CNN-LSTM architecture, per the paper's description (Section 5.1): "three 1D
convolutional layers for local feature extraction, followed by a single LSTM
layer with 8 units", ~10K total parameters.

Assumption: the paper doesn't give conv filter counts, kernel sizes, or
activation functions. The choice below (16/32/16 filters, kernel size 3, ReLU)
is ours, picked to land in the same rough parameter-count ballpark (~10K) the
paper reports -- not a stated architectural detail. Framework is PyTorch rather
than the paper's presumed Keras/TensorFlow, since TensorFlow has no Python 3.14
wheel yet; the layer-for-layer structure is the same either way.
"""

from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    def __init__(self, n_features: int, conv_channels: tuple[int, int, int] = (16, 32, 16), lstm_hidden: int = 8):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=c3, hidden_size=lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features) -> conv wants (batch, n_features, lookback)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # back to (batch, lookback, channels) for the LSTM
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, lstm_hidden)
        return self.head(last_hidden).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
