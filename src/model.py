import torch.nn as nn

from src.config import TrainingConfig


class SimpleDeepfakeClassifier(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        flattened_size = 16 * (config.image_size // 4) * (config.image_size // 4)
        self.network = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_classes),
        )

    def forward(self, x):
        return self.network(x)
