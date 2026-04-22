from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

from src.baseline_experiment import TrainConfig, validate_seed_completeness

DEFAULT_BASELINE_MODELS: Tuple[str, ...] = (
    "convmixer_256_8",
    "resnet18",
    "ast_tiny",
    "efficientnet_b0",
    "mobilenet_v2",
)
DEFAULT_BASELINE_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)


def build_output_dir(base_output_dir: Path, run_id: str) -> Path:
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id must be non-empty.")
    return Path(base_output_dir).resolve() / run_id


def make_fair_train_config(
    data_dir: Path,
    base_output_dir: Path,
    run_id: str,
    model_names: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = DEFAULT_BASELINE_SEEDS,
    include_existing_runs: bool = True,
    skip_completed_runs: bool = True,
) -> TrainConfig:
    selected_models = tuple(model_names) if model_names is not None else DEFAULT_BASELINE_MODELS
    output_dir = build_output_dir(base_output_dir=base_output_dir, run_id=run_id)

    # Fairness policy for paper comparisons:
    # - all models train from scratch
    # - one shared hyperparameter bundle across all models
    return TrainConfig(
        data_dir=str(Path(data_dir).resolve()),
        output_dir=str(output_dir),
        run_id=run_id,
        data_version=run_id,
        model_names=selected_models,
        seeds=tuple(int(seed) for seed in seeds),
        split_seed=42,
        train_ratio=0.75,
        val_ratio=0.15,
        batch_size=32,
        num_workers=4,
        num_epochs=25,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5,
        min_delta=0.02,
        ast_official_imagenet_pretrain=False,
        ast_official_audioset_pretrain=False,
        ast_official_lr=1e-3,
        ast_official_weight_decay=1e-4,
        use_model_specific_hparams=False,
        include_existing_runs=include_existing_runs,
        skip_completed_runs=skip_completed_runs,
        summary_only_config_models=True,
        require_config_match_for_skip=True,
        strict_seed_completeness=True,
    )


def filter_runs_for_run(
    runs_df: pd.DataFrame,
    model_names: Sequence[str],
    run_id: str,
) -> pd.DataFrame:
    result = runs_df.copy()
    if "run_id" in result.columns:
        result = result[result["run_id"].astype(str) == str(run_id)].copy()
    result = result[result["model"].isin(list(model_names))].copy()
    return result


def ensure_seed_completeness(
    runs_df: pd.DataFrame,
    model_names: Sequence[str],
    seeds: Sequence[int],
) -> None:
    validate_seed_completeness(
        runs_df=runs_df,
        model_names=model_names,
        seeds=seeds,
        strict=True,
    )
