"""CRNN model for CTC Morse recognition.

The architecture is intentionally compact:

- A 1D convolutional stem extracts local dit/dah / gap features and reduces
  the temporal resolution by ``2^stride_layers`` so the RNN sees a sequence
  of strokes rather than raw envelope samples.
- A bidirectional GRU adds bi-temporal context.
- A linear head emits one log-softmax distribution per output frame over
  ``vocab.num_classes`` symbols (CTC blank at index 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CrnnCtcConfig:
    """Architectural parameters for ``CrnnCtcRecognizer``."""

    num_classes: int
    conv_channels: tuple[int, ...] = (32, 64, 96)
    kernel_size: int = 5
    stride_layers: tuple[int, ...] = (2, 2)  # downsample by 4 in total
    rnn_hidden: int = 128
    rnn_layers: int = 2
    dropout: float = 0.1


class CrnnCtcRecognizer(nn.Module):
    """1D conv + BiGRU + linear CTC head."""

    def __init__(self, config: CrnnCtcConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = 1
        downsample_layers = list(config.stride_layers)
        for index, out_channels in enumerate(config.conv_channels):
            stride = downsample_layers[index] if index < len(downsample_layers) else 1
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=config.kernel_size,
                        padding=config.kernel_size // 2,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.total_stride = 1
        for stride in downsample_layers:
            self.total_stride *= max(1, int(stride))

        self.rnn = nn.GRU(
            input_size=in_channels,
            hidden_size=config.rnn_hidden,
            num_layers=config.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * config.rnn_hidden, config.num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities of shape ``(time, batch, num_classes)``.

        The PyTorch CTC loss expects a time-major log-prob tensor, so we
        keep that contract at the model boundary.
        """
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        encoded = self.encoder(inputs)  # [B, C, T']
        encoded = encoded.transpose(1, 2)  # [B, T', C]
        rnn_out, _ = self.rnn(encoded)
        logits = self.classifier(rnn_out)  # [B, T', V]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1).contiguous()  # [T', B, V]

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Map raw frame lengths to encoder-output frame lengths."""
        return torch.div(
            input_lengths + self.total_stride - 1,
            self.total_stride,
            rounding_mode="floor",
        ).clamp(min=1)
