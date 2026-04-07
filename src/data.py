from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import TrainingConfig


@dataclass
class SampleBatch:
    images: torch.Tensor
    labels: torch.Tensor


class DummyDeepfakeDataset(Dataset):
    """Temporary dataset until real preprocessing is wired in."""

    def __init__(self, config: TrainingConfig, length: int = 32) -> None:
        self.length = length
        self.image_size = config.image_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.rand(3, self.image_size, self.image_size)
        label = torch.tensor(index % 2, dtype=torch.long)
        return image, label


def create_dataloader(config: TrainingConfig) -> DataLoader:
    dataset = DummyDeepfakeDataset(config=config)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
