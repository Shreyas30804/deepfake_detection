from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts")
    frames_dir: Path = Path("artifacts/frames")
    cache_dir: Path = Path("artifacts/cache")
    image_size: int = 224
    num_frames: int = 10
    max_videos: int | None = None
    val_split: float = 0.15
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    accumulation_steps: int = 2
    label_smoothing: float = 0.05
    pos_weight: float = 1.0
    num_workers: int = 2
    use_amp: bool = True
    random_seed: int = 42

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in values.items()
        }
