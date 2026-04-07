from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = Path("data")
    image_size: int = 224
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-3
    num_classes: int = 2
    random_seed: int = 42
