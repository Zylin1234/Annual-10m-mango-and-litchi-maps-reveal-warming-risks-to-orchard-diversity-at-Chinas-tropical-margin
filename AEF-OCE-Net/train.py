from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .configuration import CODE_ROOT, Config
from .data import make_loaders
from .losses import HierarchicalSegmentationLoss
from .metrics import CLASS_NAMES, confusion, metrics_from_confusion, selection_score
from .model import build_model
from .report import save_training_figures, write_json


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def device_for_training(preferred: str = "auto") -> torch.device:
    if preferred == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if preferred == "cuda":
            raise RuntimeError("CUDA was requested but is unavailable")
        print("[Warning] CUDA is unavailable; training will run on CPU.", flush=True)
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability()
    required = f"sm_{major}{minor}"
    supported = set(torch.cuda.get_arch_list())
    if required not in supported and f"compute_{major}{minor}" not in supported:
        raise RuntimeError(
            f"GPU capability {required} is not supported by PyTorch {torch.__version__}; "
            f"supported={sorted(supported)}. Install a compatible PyTorch build, or use --device cpu only for smoke tests."
        )
    device = torch.device("cuda")
    try:
        torch.zeros(1, device=device).add_(1)
    except Exception as exc:
        raise RuntimeError("CUDA was detected but this PyTorch build cannot execute on the GPU. Install a build supporting the GPU architecture.") from exc
    return device


def autocast(cfg, device):
    return torch.autocast("cuda", dtype=torch.float16) if cfg.use_amp and device.type == "cuda" else nullcontext()


def make_scaler(cfg, device):
    try:
        return torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")


def learning_rate(cfg, epoch: int) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / cfg.warmup_epochs
    progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs - 1, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def evaluate(model, loader, criterion, cfg, device, epoch: int, collect_samples: bool = True):
    model.eval(); matrix = np.zeros((3, 3), dtype=np.int64); losses = []
    sample_matrices = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    with torch.inference_mode():
        for batch_index, (x, y, metadata) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(cfg, device):
                outputs = model(x); loss, _ = criterion(outputs, y, epoch)
            prediction = outputs["logits"].argmax(1).cpu().numpy(); truth = y.cpu().numpy()
            losses.append(float(loss)); matrix += confusion(truth, prediction, cfg.ignore_index)
            if collect_samples:
                for b, sid in enumerate(metadata["sample_id"].tolist()):
                    sample_matrices[int(sid)] += confusion(truth[b], prediction[b], cfg.ignore_index)
            if cfg.max_val_batches is not None and batch_index + 1 >= cfg.max_val_batches:
                break
    metrics = metrics_from_confusion(matrix); metrics["loss"] = float(np.mean(losses)); metrics["score"] = selection_score(metrics)
    return metrics, matrix, sample_matrices


def load_c2_backbone(model, path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    current = model.state_dict(); accepted = {}
    translations = {
        ".dw.": ".body.0.",
        ".pw.": ".body.1.",
    }
    for key, value in source.items():
        candidate = key
        if key.startswith("blocks."):
            for old, new in translations.items():
                candidate = candidate.replace(old, new)
        if (key.startswith("stem.") or key.startswith("blocks.")) and candidate in current and current[candidate].shape == value.shape:
            accepted[candidate] = value
    current.update(accepted); model.load_state_dict(current)
    return {"source": str(path), "matched_tensors": len(accepted), "eligible_prefixes": ["stem", "blocks"], "adapter_and_oce_loaded": False}


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def train(cfg: Config, run_name: str, init_c2: Path | None = None) -> Path:
    cfg.validate(); seed_all(cfg.seed); device = device_for_training(cfg.device)
    train_loader, val_loader, manifest = make_loaders(cfg)
    run_dir = cfg.output_dir / cfg.variant / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run output exists: {run_dir}. Choose a new --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg).to(device)
    init_report = load_c2_backbone(model, init_c2) if init_c2 else {"source": None, "matched_tensors": 0}
    criterion = HierarchicalSegmentationLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = make_scaler(cfg, device)
    write_json(run_dir / "resolved_config.json", cfg.to_dict())
    write_json(run_dir / "initialization.json", init_report)
    write_json(run_dir / "dataset_protocol.json", {"manifest": str(cfg.manifest_file), "train_ids": [r["sample_id"] for r in manifest if r["split"] == "train"], "val_ids": [r["sample_id"] for r in manifest if r["split"] == "val"], "spatially_independent": False})
    history, best_score, best_epoch, stale = [], -float("inf"), -1, 0
    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter(); train_loader.set_epoch(epoch); model.train()
        lr = learning_rate(cfg, epoch)
        for group in optimizer.param_groups: group["lr"] = lr
        train_losses, log_sums = [], defaultdict(float)
        for x, y, _metadata in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg, device): outputs = model(x); loss, logs = criterion(outputs, y, epoch)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
            train_losses.append(float(loss.detach()))
            for key, value in logs.items(): log_sums[key] += float(value)
        val, matrix, per_sample = evaluate(model, val_loader, criterion, cfg, device, epoch)
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": float(np.mean(train_losses)), "val_loss": val["loss"], "seconds": time.perf_counter() - epoch_start, **{k: v for k, v in val.items() if k != "loss"}}
        history.append(row); pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
        print(f"[{cfg.variant}] {epoch+1:03d}/{cfg.epochs} train={row['train_loss']:.4f} val={row['val_loss']:.4f} mIoU={row['miou']:.4f} lycheeF1={row['lychee_f1']:.4f} score={row['score']:.4f}", flush=True)
        if val["score"] > best_score:
            best_score, best_epoch, stale = val["score"], epoch + 1, 0
            payload = {"model_state_dict": model.state_dict(), "epoch": best_epoch, "metrics": val, "config": cfg.to_dict(), "variant": cfg.variant, "initialization": init_report}
            atomic_checkpoint(run_dir / "best.pt", payload)
            pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(run_dir / "best_confusion.csv", encoding="utf-8-sig")
            rows = []
            meta = {r["sample_id"]: r for r in manifest}
            for sid, sample_matrix in sorted(per_sample.items()):
                rows.append({"sample_id": sid, "county": meta[sid]["county"], "class_name": meta[sid]["class_name"], **metrics_from_confusion(sample_matrix), **{f"c{i}{j}": int(sample_matrix[i,j]) for i in range(3) for j in range(3)}})
            pd.DataFrame(rows).to_csv(run_dir / "best_per_sample_metrics.csv", index=False, encoding="utf-8-sig")
            write_json(run_dir / "best_metrics.json", val)
        else:
            stale += 1
        if epoch + 1 >= cfg.min_epochs and stale >= cfg.early_stop_patience:
            print(f"Early stop: best epoch {best_epoch}", flush=True); break
    atomic_checkpoint(run_dir / "last.pt", {"model_state_dict": model.state_dict(), "epoch": len(history), "config": cfg.to_dict(), "variant": cfg.variant})
    write_json(run_dir / "DONE.json", {"best_epoch": best_epoch, "best_score": best_score, "epochs_run": len(history), "device": str(device)})
    save_training_figures(run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train the refined AEF-OCE-Net on the original fixed fold0 split.")
    parser.add_argument("--variant", choices=("C0_baseline", "C1_adapter", "C2_adapter_oce", "C3_hierarchical", "C3_minus_adapter", "C3_minus_oce"), default="C3_hierarchical")
    parser.add_argument("--run-name", default="scratch_v1")
    parser.add_argument("--init-c2", type=Path, default=None, help="Optional C_2_MARGIN checkpoint; only shape-compatible stem/blocks are loaded.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="One train/validation batch only; never use its metrics in the paper.")
    args = parser.parse_args()
    cfg = Config(variant=args.variant, device=args.device)
    if args.epochs is not None: cfg.epochs = args.epochs; cfg.min_epochs = min(cfg.min_epochs, args.epochs)
    if args.steps_per_epoch is not None: cfg.steps_per_epoch = args.steps_per_epoch
    if args.smoke_test:
        cfg.epochs=1; cfg.min_epochs=1; cfg.early_stop_patience=1; cfg.steps_per_epoch=1
        cfg.batch_size=6; cfg.class_quota=(2,2,2); cfg.thread_workers=2; cfg.prefetch_batches=1; cfg.max_val_batches=1; cfg.use_amp=False
    print(f"Completed: {train(cfg, args.run_name, args.init_c2.resolve() if args.init_c2 else None)}")


if __name__ == "__main__":
    main()
