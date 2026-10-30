from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .configuration import Config
from .data import make_loaders
from .losses import HierarchicalSegmentationLoss
from .metrics import CLASS_NAMES, metrics_from_confusion
from .model import build_model
from .report import write_json
from .train import device_for_training, evaluate


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate one fixed-split checkpoint without augmentation.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    cfg = Config(**checkpoint["config"]); cfg.validate()
    _train, val_loader, manifest = make_loaders(cfg)
    device = device_for_training(cfg.device); model = build_model(cfg, checkpoint.get("variant", cfg.variant)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = HierarchicalSegmentationLoss(cfg).to(device)
    result, matrix, per_sample = evaluate(model, val_loader, criterion, cfg, device, int(checkpoint.get("epoch", 0)))
    output = args.output.resolve() if args.output else args.checkpoint.resolve().parent / "reevaluation"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", result)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "confusion.csv", encoding="utf-8-sig")
    meta = {r["sample_id"]: r for r in manifest}; rows = []
    for sid, sample_matrix in sorted(per_sample.items()):
        rows.append({"sample_id": sid, "county": meta[sid]["county"], "class_name": meta[sid]["class_name"], **metrics_from_confusion(sample_matrix)})
    pd.DataFrame(rows).to_csv(output / "per_sample_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"Evaluation ready: {output}")


if __name__ == "__main__":
    main()
