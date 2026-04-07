from __future__ import annotations

import json
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import TrainingConfig
from src.data import create_mtcnn, prepare_dataloaders
from src.model import DualStreamDetector


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_smooth(labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    return labels * (1 - epsilon) + 0.5 * epsilon


def _autocast_enabled(device: torch.device, config: TrainingConfig) -> bool:
    return config.use_amp and device.type == "cuda"


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    scaler: GradScaler,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_predictions: list[float] = []
    all_labels: list[float] = []
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        smoothed = label_smooth(labels, config.label_smoothing)

        with autocast(enabled=_autocast_enabled(device, config)):
            logits = model(images)
            loss = criterion(logits, smoothed) / config.accumulation_steps

        scaler.scale(loss).backward()

        should_step = (step + 1) % config.accumulation_steps == 0 or (step + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * config.accumulation_steps
        all_predictions.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    return total_loss / len(loader), roc_auc_score(all_labels, all_predictions)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device, config: TrainingConfig):
    model.eval()
    total_loss = 0.0
    all_predictions: list[float] = []
    all_labels: list[float] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=_autocast_enabled(device, config)):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_predictions.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(loader),
        "auc": roc_auc_score(all_labels, all_predictions),
        "average_precision": average_precision_score(all_labels, all_predictions),
    }


def train(config: TrainingConfig | None = None) -> dict[str, object]:
    config = config or TrainingConfig()
    config.ensure_dirs()
    set_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = create_mtcnn(device=device, image_size=config.image_size)
    train_loader, val_loader, train_videos, val_videos = prepare_dataloaders(
        config=config,
        detector=detector,
        device=device,
    )

    model = DualStreamDetector(config).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([config.pos_weight], device=device)
    )
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=_autocast_enabled(device, config))

    best_auc = 0.0
    best_path = config.output_dir / "best_model.pth"
    history: list[dict[str, float]] = []

    print(
        f"Training on {device} with {len(train_videos)} train videos and "
        f"{len(val_videos)} validation videos"
    )

    for epoch in range(1, config.epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, config
        )
        val_metrics = evaluate(model, val_loader, criterion, device, config)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_loss,
            "train_auc": train_auc,
            "val_loss": float(val_metrics["loss"]),
            "val_auc": float(val_metrics["auc"]),
            "val_average_precision": float(val_metrics["average_precision"]),
        }
        history.append(epoch_metrics)

        is_best = epoch_metrics["val_auc"] > best_auc
        if is_best:
            best_auc = epoch_metrics["val_auc"]
            state_dict = (
                model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            )
            torch.save(state_dict, best_path)

        marker = " <- best" if is_best else ""
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train loss {train_loss:.4f} auc {train_auc:.4f} | "
            f"val loss {epoch_metrics['val_loss']:.4f} auc {epoch_metrics['val_auc']:.4f}"
            f"{marker}"
        )

    summary = {
        "config": config.as_dict(),
        "best_model_path": str(best_path),
        "best_val_auc": best_auc,
        "history": history,
    }
    summary_path = config.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved training summary to {summary_path}")
    return summary


def main() -> None:
    train()


if __name__ == "__main__":
    main()
