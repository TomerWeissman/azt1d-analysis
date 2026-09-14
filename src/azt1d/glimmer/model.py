"""
Both architectures the paper evaluates, per its Section 5.1 description:
CNN-LSTM ("three 1D convolutional layers for local feature extraction,
followed by a single LSTM layer with 8 units", ~10K parameters) and
CNN-Transformer ("the same convolutional front-end, followed by a single
Transformer block with 8 attention heads, a key dimension of 16, a
feed-forward size of 256, and a dropout rate of 0.1", ~50K parameters).

Assumption (CNN-LSTM): the paper doesn't give conv filter counts, kernel
sizes, or activation functions. The choice below (16/32/16 filters, kernel
size 3, ReLU) is ours, picked to land in the same rough parameter-count
ballpark (~10K) the paper reports, not a stated architectural detail.

Assumption (CNN-Transformer): "a key dimension of 16" is ambiguous without
seeing their code, read as the per-head dimension it would give a ~128-wide
embedding whose attention block alone runs well past their own ~50K total,
read as the whole embedding width it gives implausibly tiny 2-wide attention
heads. Neither matches their disclosed ~50K parameter count on its own, so we
instead size the shared embedding dimension directly to land near that
disclosed count (d_model=64, giving head_dim=8 across 8 heads), keeping the
parts of their description we *can* match exactly: 8 heads, a feed-forward
size of 256, and a dropout rate of 0.1.

Framework is PyTorch rather than the paper's presumed Keras/TensorFlow, since
TensorFlow has no Python 3.14 wheel yet; the layer-for-layer structure is the
same either way.
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


class CNNTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        conv_channels: tuple[int, int, int] = (16, 32, 64),
        n_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
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
        self.transformer = nn.TransformerEncoderLayer(
            d_model=c3,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(c3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, lookback, c3) for the transformer block
        x = self.transformer(x)
        last_step = x[:, -1, :]  # analogous to CNNLSTM's final hidden state
        return self.head(last_step).squeeze(-1)


MODEL_CLASSES = {"cnn_lstm": CNNLSTM, "cnn_transformer": CNNTransformer}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
