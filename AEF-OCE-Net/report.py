from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_training_figures(run_dir: Path) -> None:
    history_path = run_dir / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
        axes[0].plot(history.epoch, history.train_loss, label="Train")
        axes[0].plot(history.epoch, history.val_loss, label="Validation")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="a  Optimization")
        axes[0].legend(frameon=False)
        for metric, label in (("miou", "mIoU"), ("mango_f1", "Mango F1"), ("lychee_f1", "Lychee F1")):
            axes[1].plot(history.epoch, history[metric], label=label)
        axes[1].set(xlabel="Epoch", ylabel="Validation score", ylim=(0, 1), title="b  Fixed-split validation")
        axes[1].legend(frameon=False)
        for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 400})):
            fig.savefig(run_dir / f"training_curves.{suffix}", bbox_inches="tight", **kwargs)
        plt.close(fig)
    confusion_path = run_dir / "best_confusion.csv"
    if confusion_path.exists():
        matrix = pd.read_csv(confusion_path, index_col=0).to_numpy(dtype=float)
        norm = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
        image = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        labels = ("Background", "Mango", "Lychee")
        ax.set_xticks(range(3), labels, rotation=20); ax.set_yticks(range(3), labels)
        ax.set(xlabel="Predicted", ylabel="Reference", title="Normalized confusion matrix")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i,j]:.3f}", ha="center", va="center", color="white" if norm[i,j] > .55 else "black")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
        fig.savefig(run_dir / "best_confusion.pdf", bbox_inches="tight")
        fig.savefig(run_dir / "best_confusion.png", dpi=400, bbox_inches="tight")
        plt.close(fig)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

