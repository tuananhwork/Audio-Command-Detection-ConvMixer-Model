import copy
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetV3SmallFallback(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(16, 24, stride=2),
            DepthwiseSeparableBlock(24, 24, stride=1),
            DepthwiseSeparableBlock(24, 40, stride=2),
            DepthwiseSeparableBlock(40, 40, stride=1),
            DepthwiseSeparableBlock(40, 80, stride=2),
            DepthwiseSeparableBlock(80, 80, stride=1),
            nn.Conv2d(80, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
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


def build_mobilenet_v3_small(num_classes: int) -> nn.Module:
    if tv_models is not None:
        model = tv_models.mobilenet_v3_small(weights=None)
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    print("[WARN] torchvision không khả dụng, dùng MobileNetV3SmallFallback.")
    return MobileNetV3SmallFallback(num_classes)


class ASTTiny(nn.Module):
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


def build_ast_tiny(num_classes: int) -> nn.Module:
    return ASTTiny(num_classes=num_classes)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    builders = {
        "convmixer_256_8": build_convmixer_256_8,
        "resnet18": build_resnet18,
        "mobilenet_v3_small": build_mobilenet_v3_small,
        "ast_tiny": build_ast_tiny,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model name: {model_name}")
    return builders[model_name](num_classes)


@dataclass
class TrainConfig:
    data_dir: str = "../data/features/mel"
    output_dir: str = "../data/models/baselines"
    model_names: Tuple[str, ...] = (
        "convmixer_256_8",
        "resnet18",
        "mobilenet_v3_small",
        "ast_tiny",
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


def _train_one_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    config: TrainConfig,
    run_name: str = "",
) -> Tuple[nn.Module, Dict[str, List[float]], float, int]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
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
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names, split_data = build_stratified_split(
        config.data_dir,
        split_seed=config.split_seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    runs = []
    history_store = {}

    for seed in config.seeds:
        for model_name in config.model_names:
            set_seed(seed)
            run_name = f"{model_name}|seed{seed}"
            dataloaders = _create_dataloaders(
                split_data=split_data,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                seed=seed,
            )

            model = build_model(model_name, num_classes=len(class_names)).to(device)
            n_params = count_parameters(model)
            print(f"[START] model={model_name} seed={seed} params={n_params} device={device}")

            model, history, train_seconds, best_epoch = _train_one_model(
                model=model,
                dataloaders=dataloaders,
                device=device,
                config=config,
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
                "checkpoint_path": model_ckpt,
                "report_path": report_path,
            }
            runs.append(run)
            history_store[f"{model_name}_seed{seed}"] = history

            print(
                f"[DONE] model={model_name} seed={seed} "
                f"acc={run['test_accuracy']:.2f}% macro_f1={run['macro_f1']:.4f}"
            )

    runs_df = pd.DataFrame(runs)
    runs_path = os.path.join(config.output_dir, "baseline_runs.csv")
    runs_df.to_csv(runs_path, index=False)

    summary_df = (
        runs_df.groupby("model", as_index=False)
        .agg(
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
    summary_path = os.path.join(config.output_dir, "baseline_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    split_size = {
        "train": len(split_data["train"]),
        "val": len(split_data["val"]),
        "test": len(split_data["test"]),
    }
    metadata = {
        "device": str(device),
        "classes": class_names,
        "split_size": split_size,
        "runs_csv": runs_path,
        "summary_csv": summary_path,
    }
    meta_path = os.path.join(config.output_dir, "baseline_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return runs_df, summary_df, history_store, metadata
