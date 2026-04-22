import copy
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import models as tv_models
except Exception:
    tv_models = None

try:
    from src.models.ast_official_models import ASTModel as ASTOfficialModel
except Exception:
    ASTOfficialModel = None

AST_OFFICIAL_MODEL_NAMES = {"ast_official", "ast_tiny"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AudioTransform:
    def __init__(
        self,
        time_mask_prob: float = 0.5,
        freq_mask_prob: float = 0.5,
        time_stretch_prob: float = 0.5,
    ) -> None:
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.time_stretch_prob = time_stretch_prob

    def time_mask(self, mel_spec: np.ndarray, max_mask_ratio: float = 0.1) -> np.ndarray:
        if random.random() >= self.time_mask_prob:
            return mel_spec
        x = mel_spec[0] if mel_spec.ndim == 3 else mel_spec
        n_mels, n_time = x.shape
        mask_len = max(1, int(n_time * max_mask_ratio))
        mask_len = min(mask_len, n_time)
        start = random.randint(0, max(0, n_time - mask_len))
        x[:, start : start + mask_len] = 0
        return np.expand_dims(x, axis=0)

    def freq_mask(self, mel_spec: np.ndarray, max_mask_ratio: float = 0.1) -> np.ndarray:
        if random.random() >= self.freq_mask_prob:
            return mel_spec
        x = mel_spec[0] if mel_spec.ndim == 3 else mel_spec
        n_mels, _ = x.shape
        mask_len = max(1, int(n_mels * max_mask_ratio))
        mask_len = min(mask_len, n_mels)
        start = random.randint(0, max(0, n_mels - mask_len))
        x[start : start + mask_len, :] = 0
        return np.expand_dims(x, axis=0)

    def time_stretch(self, mel_spec: np.ndarray, stretch_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        if random.random() >= self.time_stretch_prob:
            return mel_spec
        x = mel_spec[0] if mel_spec.ndim == 3 else mel_spec
        n_mels, n_time = x.shape
        stretch = random.uniform(*stretch_range)
        new_time = max(1, int(n_time * stretch))
        stretched = np.zeros((n_mels, new_time), dtype=np.float32)

        old_axis = np.arange(n_time)
        new_axis = np.linspace(0, n_time - 1, new_time)
        for i in range(n_mels):
            stretched[i] = np.interp(new_axis, old_axis, x[i])

        if new_time > n_time:
            x = stretched[:, :n_time]
        elif new_time < n_time:
            x = np.pad(stretched, ((0, 0), (0, n_time - new_time)))
        else:
            x = stretched
        return np.expand_dims(x, axis=0)

    def __call__(self, mel_spec: np.ndarray) -> np.ndarray:
        mel_spec = self.time_mask(mel_spec)
        mel_spec = self.freq_mask(mel_spec)
        mel_spec = self.time_stretch(mel_spec)
        return mel_spec


class MelDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        file_path, label = self.samples[index]
        features = np.load(file_path).astype(np.float32)
        if features.ndim == 2:
            features = np.expand_dims(features, axis=0)
        if self.transform is not None:
            features = self.transform(features)
        return torch.from_numpy(features).float(), torch.tensor(label, dtype=torch.long)


def get_classes(data_dir: str) -> List[str]:
    return sorted(
        [
            d
            for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
    )


def build_stratified_split(
    data_dir: str,
    split_seed: int = 42,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
) -> Tuple[List[str], Dict[str, List[Tuple[str, int]]]]:
    classes = get_classes(data_dir)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    rng = random.Random(split_seed)

    split_data = {"train": [], "val": [], "test": []}
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
        rng.shuffle(files)

        n_files = len(files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)

        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        split_data["train"].extend(
            [(os.path.join(class_dir, f), class_to_idx[class_name]) for f in train_files]
        )
        split_data["val"].extend(
            [(os.path.join(class_dir, f), class_to_idx[class_name]) for f in val_files]
        )
        split_data["test"].extend(
            [(os.path.join(class_dir, f), class_to_idx[class_name]) for f in test_files]
        )

    return classes, split_data


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_mel_2d(file_path: str) -> np.ndarray:
    features = np.load(file_path).astype(np.float32)
    if features.ndim == 3:
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"Expected 2D/3D mel array, got shape {features.shape} from {file_path}")
    return features


def infer_mel_shape(samples: Sequence[Tuple[str, int]]) -> Tuple[int, int]:
    if not samples:
        raise ValueError("Cannot infer mel shape from empty sample list.")
    features = _load_mel_2d(samples[0][0])
    return int(features.shape[0]), int(features.shape[1])


def compute_dataset_norm_stats(samples: Sequence[Tuple[str, int]]) -> Tuple[float, float]:
    if not samples:
        raise ValueError("Cannot compute normalization stats from empty sample list.")

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for file_path, _ in samples:
        features = _load_mel_2d(file_path).astype(np.float64)
        total_sum += float(features.sum())
        total_sq_sum += float(np.square(features).sum())
        total_count += int(features.size)

    if total_count == 0:
        raise ValueError("No mel values found while computing normalization stats.")

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 1e-12)
    std = float(np.sqrt(variance))
    return float(mean), std


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet18Fallback(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_convmixer_256_8(num_classes: int) -> nn.Module:
    dim = 256
    depth = 8
    kernel_size = 9
    patch_size = 7
    return nn.Sequential(
        nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            dim,
                            dim,
                            kernel_size=kernel_size,
                            groups=dim,
                            padding="same",
                        ),
                        nn.GELU(),
                        nn.BatchNorm2d(dim),
                    )
                ),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
            for _ in range(depth)
        ],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, num_classes),
    )


def build_resnet18(num_classes: int) -> nn.Module:
    if tv_models is not None:
        model = tv_models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    print("[WARN] torchvision không khả dụng, dùng ResNet18Fallback.")
    return ResNet18Fallback(num_classes)


def build_mobilenet_v2(num_classes: int) -> nn.Module:
    if tv_models is not None:
        model = tv_models.mobilenet_v2(weights=None)
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    print("[WARN] torchvision không khả dụng, dùng ResNet18Fallback cho mobilenet_v2.")
    return ResNet18Fallback(num_classes)


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    if tv_models is not None:
        model = tv_models.efficientnet_b0(weights=None)
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    print("[WARN] torchvision không khả dụng, dùng ResNet18Fallback cho efficientnet_b0.")
    return ResNet18Fallback(num_classes)


class ASTTiny(nn.Module):
    """Legacy lightweight transformer baseline (not official AST)."""

    def __init__(
        self,
        num_classes: int,
        input_size: Tuple[int, int] = (128, 32),
        patch_size: Tuple[int, int] = (16, 4),
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        bsz = x.shape[0]
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_ast_tiny_legacy(num_classes: int) -> nn.Module:
    return ASTTiny(num_classes=num_classes)


class ASTOfficialWrapper(nn.Module):
    """Bridge dataset tensor shape [B, 1, F, T] -> official AST input [B, T, F]."""

    def __init__(
        self,
        backbone: nn.Module,
        input_mean: float = 0.0,
        input_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_mean = input_mean
        self.input_std = max(input_std, 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape [B, 1, F, T], got {tuple(x.shape)}")
        x = x.squeeze(1).transpose(1, 2)  # [B, T, F]
        x = (x - self.input_mean) / self.input_std
        return self.backbone(x)


def build_ast_official(num_classes: int, config: "TrainConfig") -> nn.Module:
    if ASTOfficialModel is None:
        raise ImportError(
            "Official AST model is unavailable. Install dependencies: "
            "`pip install timm==0.4.5 wget` and keep src/models/ast_official_models.py."
        )

    backbone = ASTOfficialModel(
        label_dim=num_classes,
        fstride=config.ast_official_fstride,
        tstride=config.ast_official_tstride,
        input_fdim=config.ast_official_input_fdim,
        input_tdim=config.ast_official_input_tdim,
        imagenet_pretrain=config.ast_official_imagenet_pretrain,
        audioset_pretrain=config.ast_official_audioset_pretrain,
        model_size=config.ast_official_model_size,
        verbose=config.ast_official_verbose,
    )
    return ASTOfficialWrapper(
        backbone=backbone,
        input_mean=config.ast_official_norm_mean,
        input_std=config.ast_official_norm_std,
    )


def build_ast_tiny(num_classes: int, config: "TrainConfig") -> nn.Module:
    tiny_config = copy.deepcopy(config)
    tiny_config.ast_official_model_size = "tiny224"
    return build_ast_official(num_classes=num_classes, config=tiny_config)


def build_model(model_name: str, num_classes: int, config: "TrainConfig") -> nn.Module:
    builders = {
        "convmixer_256_8": build_convmixer_256_8,
        "resnet18": build_resnet18,
        "mobilenet_v2": build_mobilenet_v2,
        "efficientnet_b0": build_efficientnet_b0,
        # Keep `ast_tiny` as a compatibility alias, but point to official AST tiny224.
        "ast_tiny": lambda n: build_ast_tiny(n, config),
        "ast_tiny_legacy": build_ast_tiny_legacy,
        "ast_official": lambda n: build_ast_official(n, config),
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model name: {model_name}")
    return builders[model_name](num_classes)


@dataclass
class TrainConfig:
    data_dir: str = "../data/features/mel"
    output_dir: str = "../data/models/baselines"
    # Use run_id to isolate runs by dataset/version.
    run_id: str = "default"
    data_version: Optional[str] = None
    model_names: Tuple[str, ...] = (
        "convmixer_256_8",
        "resnet18",
        "ast_official",
    )
    seeds: Tuple[int, ...] = (42,)
    split_seed: int = 42
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    num_epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    min_delta: float = 0.02
    scheduler_factor: float = 0.2
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6
    # Official AST (YuanGongND/ast) options
    ast_official_model_size: str = "tiny224"
    ast_official_fstride: int = 10
    ast_official_tstride: int = 10
    ast_official_input_fdim: int = 128
    ast_official_input_tdim: int = 32
    ast_official_imagenet_pretrain: bool = False
    ast_official_audioset_pretrain: bool = False
    ast_official_verbose: bool = True
    # Official AST README suggests normalizing spectrogram by dataset mean/std.
    ast_official_auto_input_shape: bool = True
    ast_official_auto_norm_from_train: bool = True
    ast_official_norm_mean: Optional[float] = None
    ast_official_norm_std: Optional[float] = None
    # Keep AST-specific LR options for optional per-model tuning.
    ast_official_lr: float = 1e-3
    ast_official_weight_decay: float = 1e-4
    use_model_specific_hparams: bool = False
    # Multi-run controls:
    # - include_existing_runs: load previous baseline_runs.csv before new execution
    # - skip_completed_runs: skip only when full key matches
    #   (model, seed, run_id, config_hash, data_fingerprint, split_signature)
    # - summary_only_config_models: summary only for models in config.model_names
    include_existing_runs: bool = True
    skip_completed_runs: bool = True
    summary_only_config_models: bool = True
    require_config_match_for_skip: bool = True
    strict_seed_completeness: bool = True


def _create_dataloaders(
    split_data: Dict[str, List[Tuple[str, int]]],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Dict[str, DataLoader]:
    train_transform = AudioTransform(
        time_mask_prob=0.5,
        freq_mask_prob=0.5,
        time_stretch_prob=0.5,
    )

    train_ds = MelDataset(split_data["train"], transform=train_transform)
    val_ds = MelDataset(split_data["val"], transform=None)
    test_ds = MelDataset(split_data["test"], transform=None)

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            generator=generator,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def _resolve_ast_config(config: TrainConfig, split_data: Dict[str, List[Tuple[str, int]]]) -> TrainConfig:
    resolved = copy.deepcopy(config)
    train_samples = split_data.get("train", [])

    if resolved.ast_official_auto_input_shape:
        fdim, tdim = infer_mel_shape(train_samples)
        resolved.ast_official_input_fdim = fdim
        resolved.ast_official_input_tdim = tdim

    if resolved.ast_official_input_fdim < 16 or resolved.ast_official_input_tdim < 16:
        raise ValueError(
            "Official AST uses 16x16 patches, so input_fdim and input_tdim must be >= 16. "
            f"Got ({resolved.ast_official_input_fdim}, {resolved.ast_official_input_tdim})."
        )

    if resolved.ast_official_auto_norm_from_train:
        mean, std = compute_dataset_norm_stats(train_samples)
        resolved.ast_official_norm_mean = mean
        resolved.ast_official_norm_std = std

    if resolved.ast_official_norm_mean is None or resolved.ast_official_norm_std is None:
        raise ValueError(
            "Official AST normalization is missing. Set ast_official_norm_mean/std "
            "or enable ast_official_auto_norm_from_train=True."
        )
    if resolved.ast_official_norm_std <= 0:
        raise ValueError(f"ast_official_norm_std must be > 0, got {resolved.ast_official_norm_std}.")

    if resolved.ast_official_audioset_pretrain:
        if resolved.ast_official_model_size != "base384":
            raise ValueError("audioset_pretrain=True requires ast_official_model_size='base384'.")
        if resolved.ast_official_fstride != 10 or resolved.ast_official_tstride != 10:
            raise ValueError("audioset_pretrain=True requires fstride=tstride=10.")

    return resolved


def _get_optimizer_hparams(model_name: str, config: TrainConfig) -> Tuple[float, float]:
    if config.use_model_specific_hparams and model_name in AST_OFFICIAL_MODEL_NAMES:
        return config.ast_official_lr, config.ast_official_weight_decay
    return config.lr, config.weight_decay


def _make_data_fingerprint(data_dir: str) -> str:
    base = Path(data_dir).resolve()
    hasher = hashlib.sha256()
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")

    for cls_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        hasher.update(f"class:{cls_dir.name}\n".encode("utf-8"))
        for file_path in sorted(cls_dir.glob("*.npy")):
            stat = file_path.stat()
            rel = file_path.relative_to(base).as_posix()
            hasher.update(f"{rel}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return hasher.hexdigest()[:16]


def _make_split_signature(
    split_data: Dict[str, List[Tuple[str, int]]],
    data_dir: str,
) -> str:
    base = Path(data_dir).resolve()
    hasher = hashlib.sha256()
    for split_name in ("train", "val", "test"):
        hasher.update(f"{split_name}\n".encode("utf-8"))
        for file_path, label in split_data.get(split_name, []):
            rel = Path(file_path).resolve().relative_to(base).as_posix()
            hasher.update(f"{rel}|{label}\n".encode("utf-8"))
    return hasher.hexdigest()[:16]


def _make_model_config_hash(
    config: TrainConfig,
    model_name: str,
    data_fingerprint: str,
    split_signature: str,
) -> str:
    payload = {
        "model_name": model_name,
        "data_dir": str(Path(config.data_dir).resolve()),
        "data_fingerprint": data_fingerprint,
        "split_signature": split_signature,
        "split_seed": config.split_seed,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "num_epochs": config.num_epochs,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "patience": config.patience,
        "min_delta": config.min_delta,
        "scheduler_factor": config.scheduler_factor,
        "scheduler_patience": config.scheduler_patience,
        "scheduler_min_lr": config.scheduler_min_lr,
        "use_model_specific_hparams": config.use_model_specific_hparams,
        "ast_official_model_size": config.ast_official_model_size,
        "ast_official_fstride": config.ast_official_fstride,
        "ast_official_tstride": config.ast_official_tstride,
        "ast_official_input_fdim": config.ast_official_input_fdim,
        "ast_official_input_tdim": config.ast_official_input_tdim,
        "ast_official_imagenet_pretrain": config.ast_official_imagenet_pretrain,
        "ast_official_audioset_pretrain": config.ast_official_audioset_pretrain,
        "ast_official_auto_input_shape": config.ast_official_auto_input_shape,
        "ast_official_auto_norm_from_train": config.ast_official_auto_norm_from_train,
        "ast_official_norm_mean": config.ast_official_norm_mean,
        "ast_official_norm_std": config.ast_official_norm_std,
        "ast_official_lr": config.ast_official_lr,
        "ast_official_weight_decay": config.ast_official_weight_decay,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _make_run_key(
    model_name: str,
    seed: int,
    run_id: str,
    config_hash: str,
    data_fingerprint: str,
    split_signature: str,
) -> Tuple[str, int, str, str, str, str]:
    return (
        str(model_name),
        int(seed),
        str(run_id),
        str(config_hash),
        str(data_fingerprint),
        str(split_signature),
    )


def validate_seed_completeness(
    runs_df: pd.DataFrame,
    model_names: Sequence[str],
    seeds: Sequence[int],
    strict: bool = True,
) -> Dict[str, List[int]]:
    expected = {int(seed) for seed in seeds}
    missing: Dict[str, List[int]] = {}

    for model_name in model_names:
        if runs_df.empty:
            present = set()
        else:
            model_rows = runs_df[runs_df["model"] == model_name]
            present = {int(v) for v in model_rows["seed"].tolist()}
        missing_seeds = sorted(expected - present)
        if missing_seeds:
            missing[model_name] = missing_seeds

    if missing and strict:
        parts = [f"{model}:{seeds}" for model, seeds in missing.items()]
        raise ValueError(
            "Missing seeds for baseline comparison. "
            f"Expected {sorted(expected)}; missing -> {'; '.join(parts)}"
        )
    return missing


def _train_one_model(
    model: nn.Module,
    model_name: str,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    config: TrainConfig,
    run_name: str = "",
) -> Tuple[nn.Module, Dict[str, List[float]], float, int]:
    criterion = nn.CrossEntropyLoss()
    lr, weight_decay = _get_optimizer_hparams(model_name, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = copy.deepcopy(model.state_dict())
    start_time = time.perf_counter()

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in dataloaders["train"]:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_train_loss = train_loss / max(1, len(dataloaders["train"]))
        history["train_loss"].append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        epoch_val_loss = val_loss / max(1, len(dataloaders["val"]))
        epoch_val_acc = accuracy_score(y_true, y_pred) * 100 if y_true else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(current_lr)

        scheduler.step(epoch_val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        improved = False
        if epoch_val_acc > best_val_acc + config.min_delta:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
            improved = True
        else:
            patience_counter += 1

        prefix = f"[{run_name}] " if run_name else ""
        status = "improved" if improved else f"no_improve({patience_counter}/{config.patience})"
        print(
            f"{prefix}Epoch {epoch + 1}/{config.num_epochs} | "
            f"train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f} | "
            f"val_acc={epoch_val_acc:.2f}% | lr={current_lr:.6f}->{new_lr:.6f} | {status}"
        )

        if patience_counter >= config.patience:
            print(f"{prefix}Early stopping at epoch {epoch + 1}. Best val_acc={best_val_acc:.2f}% (epoch {best_epoch})")
            break

    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    return model, history, train_seconds, best_epoch


def _evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: Sequence[str],
    device: torch.device,
) -> Dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    losses = []
    y_true = []
    y_pred = []

    infer_start = time.perf_counter()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            losses.append(criterion(outputs, labels).item())
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    infer_seconds = time.perf_counter() - infer_start

    metrics = {
        "test_loss": float(np.mean(losses)) if losses else 0.0,
        "test_accuracy": accuracy_score(y_true, y_pred) * 100 if y_true else 0.0,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "infer_seconds": infer_seconds,
        "labels": y_true,
        "predictions": y_pred,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(class_names),
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics


def run_baseline_suite(config: TrainConfig):
    if not str(config.run_id).strip():
        raise ValueError("config.run_id must be non-empty.")

    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_path = os.path.join(config.output_dir, "baseline_runs.csv")

    class_names, split_data = build_stratified_split(
        config.data_dir,
        split_seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    data_fingerprint = _make_data_fingerprint(config.data_dir)
    split_signature = _make_split_signature(split_data=split_data, data_dir=config.data_dir)

    ast_config = config
    if any(model_name in AST_OFFICIAL_MODEL_NAMES for model_name in config.model_names):
        ast_config = _resolve_ast_config(config=config, split_data=split_data)
        print(
            "[AST] Using official AST settings: "
            f"input_fdim={ast_config.ast_official_input_fdim}, "
            f"input_tdim={ast_config.ast_official_input_tdim}, "
            f"norm_mean={ast_config.ast_official_norm_mean:.6f}, "
            f"norm_std={ast_config.ast_official_norm_std:.6f}, "
            f"lr={ast_config.ast_official_lr}, wd={ast_config.ast_official_weight_decay}"
        )

    config_hash_by_model: Dict[str, str] = {}
    for model_name in config.model_names:
        model_config = ast_config if model_name in AST_OFFICIAL_MODEL_NAMES else config
        config_hash_by_model[model_name] = _make_model_config_hash(
            config=model_config,
            model_name=model_name,
            data_fingerprint=data_fingerprint,
            split_signature=split_signature,
        )

    runs_by_key: Dict[Tuple[str, int, str, str, str, str], Dict[str, object]] = {}
    history_store = {}
    trained_count = 0
    skipped_count = 0

    if config.include_existing_runs and os.path.exists(runs_path):
        existing_df = pd.read_csv(runs_path)
        required_cols = {"model", "seed", "run_id"}
        if required_cols.issubset(set(existing_df.columns)):
            existing_df = existing_df[existing_df["run_id"].astype(str) == str(config.run_id)].copy()

            for _, row in existing_df.iterrows():
                model_name = str(row["model"])
                if model_name not in config.model_names:
                    continue

                existing_config_hash = str(row.get("config_hash", ""))
                existing_data_fingerprint = str(row.get("data_fingerprint", ""))
                existing_split_signature = str(row.get("split_signature", ""))
                if config.require_config_match_for_skip:
                    if existing_config_hash != config_hash_by_model.get(model_name, ""):
                        continue
                    if existing_data_fingerprint != data_fingerprint:
                        continue
                    if existing_split_signature != split_signature:
                        continue

                key = _make_run_key(
                    model_name=model_name,
                    seed=int(row["seed"]),
                    run_id=str(config.run_id),
                    config_hash=existing_config_hash,
                    data_fingerprint=existing_data_fingerprint,
                    split_signature=existing_split_signature,
                )
                runs_by_key[key] = row.to_dict()
            print(f"[INFO] Loaded {len(runs_by_key)} existing runs from {runs_path} for run_id={config.run_id}")
        else:
            print(
                "[WARN] Existing baseline_runs.csv has no run_id column; "
                "ignoring old cache to avoid cross-version contamination."
            )

    for seed in config.seeds:
        for model_name in config.model_names:
            run_hash = config_hash_by_model[model_name]
            key = _make_run_key(
                model_name=model_name,
                seed=int(seed),
                run_id=config.run_id,
                config_hash=run_hash,
                data_fingerprint=data_fingerprint,
                split_signature=split_signature,
            )
            if config.skip_completed_runs and key in runs_by_key:
                skipped_count += 1
                print(
                    f"[SKIP] model={model_name} seed={seed} already exists "
                    f"(run_id={config.run_id}, config_hash={run_hash})"
                )
                continue

            set_seed(seed)
            run_name = f"{model_name}|seed{seed}"
            dataloaders = _create_dataloaders(
                split_data=split_data,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                seed=seed,
            )

            model_config = ast_config if model_name in AST_OFFICIAL_MODEL_NAMES else config
            model = build_model(model_name, num_classes=len(class_names), config=model_config).to(device)
            n_params = count_parameters(model)
            run_lr, run_wd = _get_optimizer_hparams(model_name, model_config)
            print(
                f"[START] model={model_name} seed={seed} params={n_params} "
                f"device={device} lr={run_lr} wd={run_wd}"
            )

            model, history, train_seconds, best_epoch = _train_one_model(
                model=model,
                model_name=model_name,
                dataloaders=dataloaders,
                device=device,
                config=model_config,
                run_name=run_name,
            )

            metrics = _evaluate_model(
                model=model,
                dataloader=dataloaders["test"],
                class_names=class_names,
                device=device,
            )

            model_ckpt = os.path.join(config.output_dir, f"{model_name}_seed{seed}_best.pth")
            torch.save(model.state_dict(), model_ckpt)

            report_path = os.path.join(config.output_dir, f"{model_name}_seed{seed}_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(metrics["classification_report"], f, ensure_ascii=False, indent=2)

            run = {
                "model": model_name,
                "seed": seed,
                "run_id": config.run_id,
                "data_version": config.data_version or config.run_id,
                "data_dir": str(Path(config.data_dir).resolve()),
                "data_fingerprint": data_fingerprint,
                "split_signature": split_signature,
                "split_seed": config.split_seed,
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "config_hash": run_hash,
                "params": n_params,
                "best_epoch": best_epoch,
                "train_seconds": train_seconds,
                "infer_seconds": metrics["infer_seconds"],
                "test_loss": metrics["test_loss"],
                "test_accuracy": metrics["test_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "lr": run_lr,
                "weight_decay": run_wd,
                "checkpoint_path": model_ckpt,
                "report_path": report_path,
            }
            runs_by_key[key] = run
            trained_count += 1
            history_store[f"{model_name}_seed{seed}"] = history

            print(
                f"[DONE] model={model_name} seed={seed} "
                f"acc={run['test_accuracy']:.2f}% macro_f1={run['macro_f1']:.4f}"
            )

    runs_df = pd.DataFrame(list(runs_by_key.values()))
    if not runs_df.empty:
        runs_df["seed"] = runs_df["seed"].astype(int)
        runs_df = runs_df.sort_values(["model", "seed"]).reset_index(drop=True)
    runs_df.to_csv(runs_path, index=False)

    summary_input = runs_df.copy()
    if not summary_input.empty:
        summary_input = summary_input[summary_input["run_id"].astype(str) == str(config.run_id)].copy()
        summary_input = summary_input[summary_input["data_fingerprint"].astype(str) == data_fingerprint].copy()
        summary_input = summary_input[summary_input["split_signature"].astype(str) == split_signature].copy()
        if config.summary_only_config_models:
            summary_input = summary_input[summary_input["model"].isin(config.model_names)].copy()
        summary_input = summary_input[
            summary_input.apply(
                lambda row: (
                    str(row.get("model")) not in config_hash_by_model
                    or str(row.get("config_hash", "")) == config_hash_by_model.get(str(row["model"]), "")
                ),
                axis=1,
            )
        ].copy()

    if not summary_input.empty or config.strict_seed_completeness:
        validate_seed_completeness(
            runs_df=summary_input,
            model_names=config.model_names,
            seeds=config.seeds,
            strict=config.strict_seed_completeness,
        )

    summary_df = pd.DataFrame()
    if not summary_input.empty:
        summary_df = (
            summary_input.groupby("model", as_index=False)
            .agg(
                n_runs=("seed", "nunique"),
                params=("params", "mean"),
                test_accuracy_mean=("test_accuracy", "mean"),
                test_accuracy_std=("test_accuracy", "std"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                weighted_f1_mean=("weighted_f1", "mean"),
                weighted_f1_std=("weighted_f1", "std"),
                train_seconds_mean=("train_seconds", "mean"),
                infer_seconds_mean=("infer_seconds", "mean"),
            )
            .fillna(0.0)
        )
        for metric in ("test_accuracy", "macro_f1", "weighted_f1"):
            std_col = f"{metric}_std"
            ci_col = f"{metric}_ci95"
            summary_df[ci_col] = np.where(
                summary_df["n_runs"] > 1,
                1.96 * summary_df[std_col] / np.sqrt(summary_df["n_runs"]),
                0.0,
            )

    summary_path = os.path.join(config.output_dir, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    split_size = {
        "train": len(split_data["train"]),
        "val": len(split_data["val"]),
        "test": len(split_data["test"]),
    }
    metadata = {
        "device": str(device),
        "run_id": config.run_id,
        "data_version": config.data_version or config.run_id,
        "data_dir": str(Path(config.data_dir).resolve()),
        "data_fingerprint": data_fingerprint,
        "split_signature": split_signature,
        "config_hash_by_model": config_hash_by_model,
        "classes": class_names,
        "split_size": split_size,
        "requested_models": list(config.model_names),
        "requested_seeds": list(config.seeds),
        "trained_count": trained_count,
        "skipped_count": skipped_count,
        "use_model_specific_hparams": config.use_model_specific_hparams,
        "ast_official_imagenet_pretrain": config.ast_official_imagenet_pretrain,
        "strict_seed_completeness": config.strict_seed_completeness,
        "require_config_match_for_skip": config.require_config_match_for_skip,
        "include_existing_runs": config.include_existing_runs,
        "skip_completed_runs": config.skip_completed_runs,
        "summary_only_config_models": config.summary_only_config_models,
        "runs_csv": runs_path,
        "summary_csv": summary_path,
    }
    meta_path = os.path.join(config.output_dir, "baseline_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return runs_df, summary_df, history_store, metadata
