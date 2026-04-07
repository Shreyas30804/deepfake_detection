from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.config import TrainingConfig

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError as exc:  # pragma: no cover - dependency guard
    A = None
    ToTensorV2 = None
    _ALBUMENTATIONS_IMPORT_ERROR = exc
else:
    _ALBUMENTATIONS_IMPORT_ERROR = None

try:
    from facenet_pytorch import MTCNN
except ImportError as exc:  # pragma: no cover - dependency guard
    MTCNN = None
    _MTCNN_IMPORT_ERROR = exc
else:
    _MTCNN_IMPORT_ERROR = None


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class FaceExtractionResult:
    img_path: str
    label: int
    video_id: str
    split: str


def _require_albumentations() -> None:
    if A is None or ToTensorV2 is None:
        raise ImportError(
            "albumentations and albumentations.pytorch are required for dataset transforms"
        ) from _ALBUMENTATIONS_IMPORT_ERROR


def _require_mtcnn() -> None:
    if MTCNN is None:
        raise ImportError(
            "facenet-pytorch is required for face extraction"
        ) from _MTCNN_IMPORT_ERROR


def build_transforms() -> tuple[object, object]:
    _require_albumentations()
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.4
            ),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            A.ImageCompression(quality_range=(50, 95), p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.3
            ),
            A.CoarseDropout(max_holes=4, max_height=20, max_width=20, p=0.2),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    return train_transform, val_transform


class DeepfakeFaceDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_size: int, transform=None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image = cv2.imread(row["img_path"])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {row['img_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        if self.transform is None:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            image = self.transform(image=image)["image"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label


def discover_videos(config: TrainingConfig) -> pd.DataFrame:
    meta_files = sorted(config.data_dir.glob("**/metadata.json"))
    if not meta_files:
        raise FileNotFoundError(
            f"No metadata.json files found under {config.data_dir}. "
            "Point TrainingConfig.data_dir at a DFDC-style dataset root."
        )

    records: list[dict[str, object]] = []
    for meta_file in meta_files:
        with meta_file.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        part_dir = meta_file.parent
        for filename, info in metadata.items():
            video_path = part_dir / filename
            if not video_path.exists():
                continue
            records.append(
                {
                    "video_id": video_path.stem,
                    "path": str(video_path),
                    "label": 0 if info["label"] == "REAL" else 1,
                }
            )

    video_df = pd.DataFrame(records).drop_duplicates(subset=["video_id"]).reset_index(drop=True)
    if video_df.empty:
        raise ValueError("Dataset metadata was found, but no videos were collected.")

    if config.max_videos:
        per_class_cap = max(config.max_videos // 2, 1)
        video_df = (
            video_df.groupby("label", group_keys=False)
            .apply(
                lambda group: group.sample(
                    min(len(group), per_class_cap), random_state=config.random_seed
                )
            )
            .reset_index(drop=True)
        )
    return video_df


def split_videos(video_df: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        video_df,
        test_size=config.val_split,
        stratify=video_df["label"],
        random_state=config.random_seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _cache_key(config: TrainingConfig) -> str:
    payload = json.dumps(
        {
            "image_size": config.image_size,
            "num_frames": config.num_frames,
            "max_videos": config.max_videos,
            "random_seed": config.random_seed,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def _cache_path(config: TrainingConfig, split_name: str) -> Path:
    return config.cache_dir / f"{split_name}_faces_{_cache_key(config)}.csv"


def create_mtcnn(device: torch.device, image_size: int) -> object:
    _require_mtcnn()
    return MTCNN(keep_all=True, device=device, image_size=image_size)


def extract_frames(video_path: str | Path, n_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return []

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def extract_face(frame_rgb: np.ndarray, detector, image_size: int, margin: float = 0.15) -> np.ndarray | None:
    boxes, probs = detector.detect(Image.fromarray(frame_rgb))
    if boxes is None or probs is None:
        return None

    best_index = int(np.argmax(probs))
    if probs[best_index] is None or float(probs[best_index]) <= 0:
        return None

    x1, y1, x2, y2 = boxes[best_index]
    height, width = frame_rgb.shape[:2]
    box_width = x2 - x1
    box_height = y2 - y1
    x_margin = box_width * margin
    y_margin = box_height * margin

    left = max(int(x1 - x_margin), 0)
    top = max(int(y1 - y_margin), 0)
    right = min(int(x2 + x_margin), width)
    bottom = min(int(y2 + y_margin), height)
    if right <= left or bottom <= top:
        return None

    face = frame_rgb[top:bottom, left:right]
    if face.size == 0:
        return None
    return cv2.resize(face, (image_size, image_size))


def process_video(
    video_path: str | Path,
    detector,
    config: TrainingConfig,
    split_name: str,
    label: int,
    video_id: str,
) -> list[FaceExtractionResult]:
    split_dir = config.frames_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    results: list[FaceExtractionResult] = []
    for frame_index, frame in enumerate(extract_frames(video_path, config.num_frames)):
        face = extract_face(frame, detector, config.image_size)
        if face is None:
            continue
        output_path = split_dir / f"{video_id}_{frame_index:02d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        results.append(
            FaceExtractionResult(
                img_path=str(output_path),
                label=label,
                video_id=video_id,
                split=split_name,
            )
        )
    return results


def materialize_face_cache(
    video_df: pd.DataFrame,
    split_name: str,
    config: TrainingConfig,
    detector,
) -> pd.DataFrame:
    cache_path = _cache_path(config, split_name)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    records: list[dict[str, object]] = []
    for row in video_df.itertuples(index=False):
        extracted = process_video(
            video_path=row.path,
            detector=detector,
            config=config,
            split_name=split_name,
            label=int(row.label),
            video_id=str(row.video_id),
        )
        for item in extracted:
            records.append(
                {
                    "img_path": item.img_path,
                    "label": item.label,
                    "video_id": item.video_id,
                    "split": item.split,
                }
            )

    face_df = pd.DataFrame(records)
    face_df.to_csv(cache_path, index=False)
    return face_df


def prepare_dataloaders(
    config: TrainingConfig, detector, device: torch.device
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    del device
    config.ensure_dirs()
    video_df = discover_videos(config)
    train_videos, val_videos = split_videos(video_df, config)
    train_faces = materialize_face_cache(train_videos, "train", config, detector)
    val_faces = materialize_face_cache(val_videos, "val", config, detector)

    if train_faces.empty or val_faces.empty:
        raise ValueError("Face extraction produced an empty train or validation set.")

    train_transform, val_transform = build_transforms()
    train_dataset = DeepfakeFaceDataset(
        train_faces, image_size=config.image_size, transform=train_transform
    )
    val_dataset = DeepfakeFaceDataset(
        val_faces, image_size=config.image_size, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_videos, val_videos
