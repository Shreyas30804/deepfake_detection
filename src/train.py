from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from src.config import TrainingConfig
from src.data import create_dataloader
from src.model import SimpleDeepfakeClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def main() -> None:
    config = TrainingConfig()
    set_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = create_dataloader(config)
    model = SimpleDeepfakeClassifier(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device} for {config.epochs} epoch(s)")

    for epoch in range(config.epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{config.epochs} - loss: {loss:.4f}")


if __name__ == "__main__":
    main()
