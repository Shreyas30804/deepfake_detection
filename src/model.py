from __future__ import annotations

import torch
import torch.nn as nn

from src.config import TrainingConfig

try:
    import timm
except ImportError as exc:  # pragma: no cover - dependency guard
    timm = None
    _TIMM_IMPORT_ERROR = exc
else:
    _TIMM_IMPORT_ERROR = None


def _require_timm() -> None:
    if timm is None:
        raise ImportError("timm is required to build the detector backbone") from _TIMM_IMPORT_ERROR


class FrequencyStream(nn.Module):
    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(inputs)
        shifted = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(shifted) + 1e-8)
        mag_min = magnitude.amin(dim=(2, 3), keepdim=True)
        mag_max = magnitude.amax(dim=(2, 3), keepdim=True)
        normalized = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
        return self.network(normalized)


class DualStreamDetector(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        _require_timm()
        self.spatial = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        spatial_features = self.spatial.num_features
        self.frequency = FrequencyStream(out_features=128)
        self.classifier = nn.Sequential(
            nn.Linear(spatial_features + 128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(config.dropout / 2),
            nn.Linear(128, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial(inputs)
        frequency_features = self.frequency(inputs)
        fused = torch.cat([spatial_features, frequency_features], dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(1)
