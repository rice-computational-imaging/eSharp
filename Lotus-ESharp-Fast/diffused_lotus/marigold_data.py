from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _import_marigold(marigold_root: Path):
    root_str = str(marigold_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from src.dataset import DatasetMode, get_dataset
    from src.util.config_util import recursive_load_config
    from src.util.depth_transform import get_depth_normalizer

    return DatasetMode, get_dataset, recursive_load_config, get_depth_normalizer


def _resolve_split_filenames(cfg: DictConfig, marigold_root: Path) -> None:
    split_configs: list[Any] = [cfg.dataset.train]
    split_configs.extend(list(cfg.dataset.get("val", [])))
    split_configs.extend(list(cfg.dataset.get("vis", [])))

    for split_cfg in split_configs:
        filenames = split_cfg.get("filenames")
        if filenames is None:
            continue
        filenames_path = Path(str(filenames))
        if not filenames_path.is_absolute():
            split_cfg.filenames = str((marigold_root / filenames_path).resolve())


def load_marigold_config(marigold_root: str | Path, config_path: str | Path) -> DictConfig:
    marigold_root = Path(marigold_root).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = marigold_root / config_path

    if not marigold_root.is_dir():
        raise FileNotFoundError(f"Marigold root does not exist: {marigold_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Marigold config does not exist: {config_path}")

    _, _, recursive_load_config, _ = _import_marigold(marigold_root)
    with _working_directory(marigold_root):
        cfg = recursive_load_config(str(config_path))
    _resolve_split_filenames(cfg, marigold_root)
    return cfg


def build_train_dataset(
    marigold_root: str | Path,
    marigold_config: str | Path,
    base_data_dir: str | Path,
):
    marigold_root = Path(marigold_root).resolve()
    base_data_dir = Path(base_data_dir).resolve()
    DatasetMode, get_dataset, _, get_depth_normalizer = _import_marigold(marigold_root)
    cfg = load_marigold_config(marigold_root, marigold_config)

    train_cfg = cfg.dataset.train
    if train_cfg.name in {"mixed", "multi_cycle"}:
        raise NotImplementedError(
            "The Lotus bridge currently accepts one Marigold training dataset per run."
        )

    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    event_input_mode = OmegaConf.select(
        cfg, "dataset.train.event_input_mode", default="tenacode"
    )
    event_kwargs = (
        {}
        if train_cfg.get("event_input_mode") is not None
        else {"event_input_mode": event_input_mode}
    )
    dataset = get_dataset(
        train_cfg,
        base_data_dir=str(base_data_dir),
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
        **event_kwargs,
    )
    return dataset, cfg


def build_train_dataloader(
    marigold_root: str | Path,
    marigold_config: str | Path,
    base_data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DictConfig]:
    dataset, cfg = build_train_dataset(
        marigold_root=marigold_root,
        marigold_config=marigold_config,
        base_data_dir=base_data_dir,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    return loader, cfg


def build_validation_dataloaders(
    marigold_root: str | Path,
    marigold_config: str | Path,
    base_data_dir: str | Path,
    num_workers: int,
):
    marigold_root = Path(marigold_root).resolve()
    base_data_dir = Path(base_data_dir).resolve()
    DatasetMode, get_dataset, _, get_depth_normalizer = _import_marigold(marigold_root)
    cfg = load_marigold_config(marigold_root, marigold_config)
    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    train_event_mode = OmegaConf.select(
        cfg, "dataset.train.event_input_mode", default="tenacode"
    )

    loaders = []
    for split_cfg in cfg.dataset.val:
        event_kwargs = (
            {}
            if split_cfg.get("event_input_mode") is not None
            else {"event_input_mode": train_event_mode}
        )
        dataset = get_dataset(
            split_cfg,
            base_data_dir=str(base_data_dir),
            mode=DatasetMode.EVAL,
            augmentation_args=cfg.augmentation,
            depth_transform=depth_transform,
            **event_kwargs,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )
        )
    return loaders, depth_transform, cfg


def build_visualization_dataloaders(
    marigold_root: str | Path,
    marigold_config: str | Path,
    base_data_dir: str | Path,
    num_workers: int,
):
    """Build the original Marigold ``dataset.vis`` loaders for qualitative checks."""
    marigold_root = Path(marigold_root).resolve()
    base_data_dir = Path(base_data_dir).resolve()
    DatasetMode, get_dataset, _, get_depth_normalizer = _import_marigold(marigold_root)
    cfg = load_marigold_config(marigold_root, marigold_config)
    depth_transform = get_depth_normalizer(cfg.depth_normalization)
    train_event_mode = OmegaConf.select(
        cfg, "dataset.train.event_input_mode", default="tenacode"
    )

    loaders = []
    for split_cfg in cfg.dataset.get("vis", []):
        event_kwargs = (
            {}
            if split_cfg.get("event_input_mode") is not None
            else {"event_input_mode": train_event_mode}
        )
        dataset = get_dataset(
            split_cfg,
            base_data_dir=str(base_data_dir),
            mode=DatasetMode.EVAL,
            augmentation_args=cfg.augmentation,
            depth_transform=depth_transform,
            **event_kwargs,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )
        )
    return loaders
