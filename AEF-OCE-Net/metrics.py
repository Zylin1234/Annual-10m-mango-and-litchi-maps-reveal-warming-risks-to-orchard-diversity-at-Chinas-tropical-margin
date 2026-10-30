from __future__ import annotations

import numpy as np


CLASS_NAMES = ("background", "mango", "lychee")


def confusion(target, prediction, ignore_index: int = 255) -> np.ndarray:
    target, prediction = np.asarray(target).ravel(), np.asarray(prediction).ravel()
    valid = target != ignore_index
    values = 3 * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(values, minlength=9).reshape(3, 3)


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    x = matrix.astype(np.float64)
    tp, truth, predicted = np.diag(x), x.sum(1), x.sum(0)
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(truth, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    iou = tp / np.maximum(truth + predicted - tp, 1)
    total = x.sum()
    oa = tp.sum() / max(total, 1)
    pe = (truth * predicted).sum() / max(total * total, 1)
    result = {"oa": oa, "kappa": (oa - pe) / max(1 - pe, 1e-12), "miou": iou.mean()}
    for i, name in enumerate(CLASS_NAMES):
        result.update({f"{name}_precision": precision[i], f"{name}_recall": recall[i], f"{name}_f1": f1[i], f"{name}_iou": iou[i]})
    result["orchard_to_background"] = (x[1:, 0].sum() / max(x[1:].sum(), 1))
    result["mango_to_lychee"] = x[1, 2] / max(x[1].sum(), 1)
    result["lychee_to_mango"] = x[2, 1] / max(x[2].sum(), 1)
    return {key: float(value) for key, value in result.items()}


def selection_score(metrics: dict[str, float]) -> float:
    return 0.50 * metrics["miou"] + 0.25 * metrics["mango_f1"] + 0.25 * metrics["lychee_f1"]

