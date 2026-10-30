from __future__ import annotations

import csv
import math
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        row["patches"] = int(row["patches"])
    return rows


class FixedSplitZarrDataset(Dataset):
    def __init__(self, root: Path, manifest: list[dict], split: str, augment: bool = False):
        import pandas as pd

        if split not in {"train", "val"}:
            raise ValueError(split)
        self.root, self.split, self.augment = Path(root), split, augment
        self.meta = {r["sample_id"]: r for r in manifest}
        index = pd.read_csv(self.root / "index.csv")
        index = index[index["split"] == split].sort_values("sample_id")
        self.sample_ids, self.local_indices, self.sample_classes = [], [], []
        for row in index.itertuples(index=False):
            for local in range(int(row.patches)):
                self.sample_ids.append(int(row.sample_id))
                self.local_indices.append(local)
                self.sample_classes.append(str(row.class_name))
        expected = sum(r["patches"] for r in manifest if r["split"] == split)
        if len(self.sample_ids) != expected:
            raise ValueError(f"{split}: index contains {len(self.sample_ids)} patches, expected {expected}")
        self._thread_state = threading.local()

    def __len__(self):
        return len(self.sample_ids)

    def _group(self, sid: int):
        import zarr

        groups = getattr(self._thread_state, "groups", None)
        if groups is None:
            groups = {}; self._thread_state.groups = groups
        if sid not in groups:
            groups[sid] = zarr.open_group(str(self.root / f"ID{sid:02d}.zarr"), mode="r")["patches"]
        return groups[sid]

    def get_item(self, index: int, augment_seed: int | None = None):
        sid, local = self.sample_ids[index], self.local_indices[index]
        group = self._group(sid)
        image = np.asarray(group["image"][local], dtype=np.float32)
        label = np.asarray(group["label"][local], dtype=np.uint8).astype(np.int64)
        finite = np.all(np.isfinite(image), axis=0)
        aef_present = ~np.all(image <= -127.5, axis=0)
        input_valid = (finite & aef_present).astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image[:, input_valid == 0] = 0.0
        if self.augment:
            rng = np.random.default_rng(augment_seed if augment_seed is not None else (sid * 1000003 + local))
            if rng.random() < 0.5:
                image, label, input_valid = image[:, :, ::-1].copy(), label[:, ::-1].copy(), input_valid[:, ::-1].copy()
            if rng.random() < 0.5:
                image, label, input_valid = image[:, ::-1, :].copy(), label[::-1, :].copy(), input_valid[::-1, :].copy()
            k = rng.integers(4)
            if k:
                image = np.rot90(image, k, (1, 2)).copy(); label = np.rot90(label, k).copy(); input_valid = np.rot90(input_valid, k).copy()
        x = np.concatenate((image, input_valid[None]), axis=0)
        row = self.meta[sid]
        metadata = {"sample_id": sid, "county": row["county"], "sample_class": row["class_name"], "store_index": local}
        return torch.from_numpy(x), torch.from_numpy(label), metadata

    def __getitem__(self, index: int):
        return self.get_item(index)


class BalancedBatchPlan:
    ORDER = ("background", "mango", "lychee")

    def __init__(self, dataset: FixedSplitZarrDataset, quota: tuple[int, int, int], batches: int, seed: int):
        self.dataset, self.quota, self.batches, self.seed, self.epoch = dataset, quota, batches, seed, 0
        groups = defaultdict(list)
        for i, name in enumerate(dataset.sample_classes):
            groups[name].append(i)
        self.groups = {name: torch.tensor(groups[name], dtype=torch.long) for name in self.ORDER}
        if any(len(self.groups[name]) == 0 for name in self.ORDER):
            raise ValueError("All three classes must be represented in training")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def indices(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1009)
        result = []
        for _ in range(self.batches):
            batch = []
            for name, count in zip(self.ORDER, self.quota):
                pool = self.groups[name]
                batch.extend(pool[torch.randint(len(pool), (count,), generator=generator)].tolist())
            permutation = torch.randperm(len(batch), generator=generator).tolist()
            result.append([batch[i] for i in permutation])
        return result


class ThreadBatchLoader:
    def __init__(self, dataset, batch_size: int, workers: int, prefetch: int, plan: BalancedBatchPlan | None = None):
        self.dataset, self.batch_size, self.plan = dataset, batch_size, plan
        self.workers, self.prefetch = max(1, workers), max(1, prefetch)
        self.epoch = 0

    def __len__(self):
        return self.plan.batches if self.plan else math.ceil(len(self.dataset) / self.batch_size)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.plan:
            self.plan.set_epoch(epoch)

    def _batches(self):
        if self.plan:
            return self.plan.indices()
        ids = list(range(len(self.dataset)))
        return [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

    def _load_batch(self, batch_id: int, indices: list[int]):
        base = (self.plan.seed if self.plan else 0) + self.epoch * 1000003 + batch_id * 10007
        return default_collate([self.dataset.get_item(index, base + position) for position, index in enumerate(indices)])

    def __iter__(self):
        batches = iter(enumerate(self._batches()))
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="aef_zarr") as executor:
            pending = deque()
            for _ in range(self.prefetch):
                try:
                    batch_id, indices = next(batches)
                except StopIteration:
                    break
                pending.append(executor.submit(self._load_batch, batch_id, indices))
            while pending:
                future = pending.popleft()
                try:
                    batch_id, indices = next(batches)
                except StopIteration:
                    indices = None
                if indices is not None:
                    pending.append(executor.submit(self._load_batch, batch_id, indices))
                yield future.result()


def make_loaders(cfg):
    manifest = read_manifest(cfg.manifest_file)
    train = FixedSplitZarrDataset(cfg.data_dir, manifest, "train", augment=True)
    val = FixedSplitZarrDataset(cfg.data_dir, manifest, "val", augment=False)
    plan = BalancedBatchPlan(train, cfg.class_quota, cfg.steps_per_epoch, cfg.seed)
    train_loader = ThreadBatchLoader(train, cfg.batch_size, cfg.thread_workers, cfg.prefetch_batches, plan)
    val_loader = ThreadBatchLoader(val, cfg.batch_size, cfg.thread_workers, cfg.prefetch_batches)
    return train_loader, val_loader, manifest
