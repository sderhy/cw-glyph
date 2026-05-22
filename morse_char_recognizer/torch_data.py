"""PyTorch dataset adapters for synthetic Morse-character examples."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

from morse_char_recognizer.dataset import SyntheticDatasetConfig, SyntheticMorseCharacterDataset


class TorchMorseCharacterDataset(Dataset):
    """Return ``(envelope[1, time], label)`` tensors from the lazy generator."""

    def __init__(self, config: SyntheticDatasetConfig) -> None:
        self.base = SyntheticMorseCharacterDataset(config)

    @property
    def classes(self) -> tuple[str, ...]:
        return self.base.classes

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.base[index]
        x = torch.from_numpy(example.envelope).unsqueeze(0).float()
        y = torch.tensor(example.label, dtype=torch.long)
        return x, y


class CachedTorchMorseCharacterDataset(Dataset):
    """Precompute synthetic examples once for faster multi-epoch training."""

    def __init__(self, config: SyntheticDatasetConfig) -> None:
        base = SyntheticMorseCharacterDataset(config)
        xs: list[Tensor] = []
        ys: list[int] = []
        for example in base:
            xs.append(torch.from_numpy(example.envelope).unsqueeze(0).float())
            ys.append(example.label)
        self.x = torch.stack(xs)
        self.y = torch.tensor(ys, dtype=torch.long)
        self.classes = base.classes

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
