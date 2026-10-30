from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT.parents[1]


@dataclass
class Config:
    source_npz: str = str(CODE_ROOT / "01-processed_cv")
    data_path: str = str(PROJECT / "data" / "patch_store_zarr_v1")
    manifest_path: str = str(PROJECT / "config" / "fixed_split_manifest.csv")
    output_root: str = str(PROJECT / "outputs" / "fixed_split")

    variant: str = "C3_hierarchical"
    in_channels: int = 64
    num_classes: int = 3
    base_dim: int = 64
    adapter_rank: int = 16
    num_blocks: int = 3
    dropout: float = 0.12
    ignore_index: int = 255

    batch_size: int = 24
    class_quota: tuple[int, int, int] = (6, 9, 9)  # background, mango, lychee
    steps_per_epoch: int = 240
    thread_workers: int = 4
    prefetch_batches: int = 4
    max_val_batches: int | None = None
    epochs: int = 80
    min_epochs: int = 30
    early_stop_patience: int = 12
    lr: float = 4e-4
    min_lr: float = 2e-5
    warmup_epochs: int = 5
    weight_decay: float = 7e-4
    grad_clip: float = 1.0
    seed: int = 2026
    use_amp: bool = True
    device: str = "auto"

    focal_alpha: tuple[float, float, float] = (0.45, 1.15, 1.18)
    focal_gamma: float = 1.5
    tversky_alpha: float = 0.40
    tversky_beta: float = 0.60
    lambda_tversky: float = 0.40
    lambda_orchard_aux: float = 0.15
    lambda_species_aux: float = 0.20
    lambda_species_margin: float = 0.06
    species_margin: float = 0.20
    margin_topk_ratio: float = 0.35
    margin_start_epoch: int = 4
    margin_ramp_epochs: int = 8

    @property
    def data_dir(self) -> Path:
        return Path(self.data_path)

    @property
    def manifest_file(self) -> Path:
        return Path(self.manifest_path)

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root)

    def validate(self) -> None:
        if self.variant not in {"C0_baseline", "C1_adapter", "C2_adapter_oce", "C3_hierarchical", "C3_minus_adapter", "C3_minus_oce"}:
            raise ValueError(f"Unknown variant: {self.variant}")
        if sum(self.class_quota) != self.batch_size:
            raise ValueError("class_quota must sum to batch_size")
        if self.in_channels != 64 or self.num_classes != 3:
            raise ValueError("This study protocol requires 64 AEF dimensions and three classes")
        if not 0 < self.margin_topk_ratio <= 1:
            raise ValueError("margin_topk_ratio must be in (0, 1]")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if self.max_val_batches is not None and self.max_val_batches < 1:
            raise ValueError("max_val_batches must be None or positive")

    def to_dict(self) -> dict:
        return asdict(self)
