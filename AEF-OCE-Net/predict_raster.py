from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

from .configuration import Config
from .model import build_model
from .train import device_for_training


def patch_probabilities(model, image: np.ndarray, device, patch: int, stride: int, batch_size: int) -> np.ndarray:
    channels, height, width = image.shape
    out_h = max(patch, int(math.ceil(max(height - patch, 0) / stride) * stride + patch))
    out_w = max(patch, int(math.ceil(max(width - patch, 0) / stride) * stride + patch))
    padded = np.full((channels, out_h, out_w), -128.0, dtype=np.float32); padded[:, :height, :width] = image
    finite = np.all(np.isfinite(padded[:64]), axis=0)
    present = ~np.all(padded[:64] <= -127.5, axis=0)
    valid = (finite & present).astype(np.float32)
    padded[:64] = np.nan_to_num(padded[:64], nan=0.0, posinf=0.0, neginf=0.0); padded[:64, valid == 0] = 0
    network_input = np.concatenate((padded[:64], valid[None]), axis=0)
    accumulator = np.zeros((3, out_h, out_w), dtype=np.float32); weights = np.zeros((out_h, out_w), dtype=np.float32)
    axis = np.hanning(patch).astype(np.float32); blend = np.maximum(np.outer(axis, axis), 1e-3)
    positions = [(y, x) for y in range(0, out_h - patch + 1, stride) for x in range(0, out_w - patch + 1, stride)]
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            batch_pos = positions[start : start + batch_size]
            batch = torch.from_numpy(np.stack([network_input[:, y:y+patch, x:x+patch] for y, x in batch_pos])).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                prob = torch.softmax(model(batch)["logits"], 1).float().cpu().numpy()
            for values, (y, x) in zip(prob, batch_pos):
                accumulator[:, y:y+patch, x:x+patch] += values * blend
                weights[y:y+patch, x:x+patch] += blend
    probability = accumulator / np.maximum(weights, 1e-6)
    probability[:, valid == 0] = 0
    return probability[:, :height, :width]


def main():
    parser = argparse.ArgumentParser(description="Blockwise province/sample inference for one 64-band annual AEF raster.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("aef_raster", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--write-probability", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    cfg = Config(**checkpoint["config"]); device = device_for_training(cfg.device)
    model = build_model(cfg, checkpoint.get("variant", cfg.variant)).to(device); model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); pixels = 0
    with rasterio.open(args.aef_raster.resolve()) as src:
        if src.count != 64: raise ValueError(f"Expected 64 AEF bands, found {src.count}")
        class_profile = src.profile.copy(); class_profile.update(count=1, dtype="uint8", nodata=255, compress="deflate", tiled=True)
        prob_profile = src.profile.copy(); prob_profile.update(count=3, dtype="float32", nodata=-9999.0, compress="deflate", tiled=True)
        class_path = output / "AEF_OCE_class.tif"; probability_path = output / "AEF_OCE_probability.tif"
        probability_writer = rasterio.open(probability_path, "w", **prob_profile) if args.write_probability else None
        with rasterio.open(class_path, "w", **class_profile) as class_writer:
            try:
                halo = max(args.patch - args.stride, args.patch // 2)
                for row in range(0, src.height, args.block):
                    for col in range(0, src.width, args.block):
                        h, w = min(args.block, src.height-row), min(args.block, src.width-col)
                        r0, c0 = max(0, row-halo), max(0, col-halo)
                        r1, c1 = min(src.height, row+h+halo), min(src.width, col+w+halo)
                        read_window = Window(c0, r0, c1-c0, r1-r0)
                        write_window = Window(col, row, w, h)
                        image = src.read(window=read_window, out_dtype="float32")
                        expanded = patch_probabilities(model, image, device, args.patch, args.stride, args.batch_size)
                        y0, x0 = row-r0, col-c0
                        probability = expanded[:, y0:y0+h, x0:x0+w]
                        valid = probability.sum(0) > 0; classes = probability.argmax(0).astype(np.uint8); classes[~valid] = 255
                        class_writer.write(classes, 1, window=write_window)
                        if probability_writer:
                            values = probability.astype(np.float32); values[:, ~valid] = -9999.0; probability_writer.write(values, window=write_window)
                        pixels += h*w; print(f"row={row}, col={col}", flush=True)
            finally:
                if probability_writer: probability_writer.close()
    elapsed = time.perf_counter()-started
    (output / "inference_report.json").write_text(json.dumps({"seconds":elapsed,"pixels":pixels,"pixels_per_second":pixels/max(elapsed,1e-9),"checkpoint":str(args.checkpoint.resolve()),"source":str(args.aef_raster.resolve())}, indent=2), encoding="utf-8")
    print(f"Inference ready: {output}")


if __name__ == "__main__":
    main()
