from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from .configuration import Config


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sample_id"] = int(row["sample_id"])
        row["patches"] = int(row["patches"])
    return rows


def source_files(source: Path, row: dict) -> list[Path]:
    folder = source / "fold0" / row["split"] / row["class_name"]
    files = sorted(folder.glob(f"ID{row['sample_id']}_*.npz"))
    if len(files) != row["patches"]:
        raise ValueError(
            f"ID{row['sample_id']} {row['split']}: manifest expects {row['patches']} patches, found {len(files)} in {folder}"
        )
    return files


def codec(level: int):
    return Blosc(cname="zstd", clevel=level, shuffle=Blosc.NOSHUFFLE)


def _scalar(item, key: str, default: int) -> int:
    return int(np.asarray(item[key]).item()) if key in item else int(default)


def build_sample(output: Path, source: Path, row: dict, compressor, write_batch: int) -> list[dict]:
    sid = row["sample_id"]
    destination = output / f"ID{sid:02d}.zarr"
    if destination.exists():
        root = zarr.open_group(str(destination), mode="r")
        if bool(root.attrs.get("complete", False)) and int(root.attrs.get("patches", -1)) == row["patches"]:
            print(f"[Skip] ID{sid}: complete", flush=True)
            return []
        raise FileExistsError(f"Incomplete or incompatible store exists: {destination}")
    files = source_files(source, row)
    # DirectoryStore renames are intermittently denied on Windows even after
    # closing all Python handles (typically an antivirus/indexer race).  Build
    # at the final per-sample path and use the root `complete` attribute as the
    # transaction marker.  Readers reject stores without complete=True.
    root = zarr.open_group(str(destination), mode="w")
    root.attrs.update(
        format="AEF_FIXED_SPLIT_ZARR_V1",
        sample_id=sid,
        class_name=row["class_name"],
        county=row["county"],
        split=row["split"],
        patches=len(files),
        complete=False,
    )
    group = root.create_group("patches")
    n = len(files)
    image_array = group.create_dataset("image", shape=(n, 64, 128, 128), chunks=(1, 64, 128, 128), dtype="f4", compressor=compressor)
    label_array = group.create_dataset("label", shape=(n, 128, 128), chunks=(1, 128, 128), dtype="u1", compressor=compressor)
    x_array = group.create_dataset("x", shape=(n,), chunks=(min(1024, n),), dtype="i4", compressor=compressor)
    y_array = group.create_dataset("y", shape=(n,), chunks=(min(1024, n),), dtype="i4", compressor=compressor)
    records = []
    try:
        for start in range(0, n, write_batch):
            batch = files[start : start + write_batch]
            images, labels, xs, ys = [], [], [], []
            for local, path in enumerate(batch, start=start):
                with np.load(path, allow_pickle=False) as item:
                    image = np.asarray(item["image"], dtype=np.float32)
                    label = np.asarray(item["label"], dtype=np.uint8)
                    if image.shape != (64, 128, 128) or label.shape != (128, 128):
                        raise ValueError(f"Unexpected arrays in {path}: {image.shape}, {label.shape}")
                    embedded_sid = _scalar(item, "sample_id", sid)
                    if embedded_sid != sid:
                        raise ValueError(f"{path}: embedded ID{embedded_sid} != filename ID{sid}")
                    x, y = _scalar(item, "x", -1), _scalar(item, "y", -1)
                images.append(image)
                labels.append(label)
                xs.append(x)
                ys.append(y)
                records.append(
                    dict(sample_id=sid, split=row["split"], class_name=row["class_name"], store_index=local, source_file=path.name, x=x, y=y)
                )
            stop = start + len(batch)
            image_array[start:stop] = np.stack(images)
            label_array[start:stop] = np.stack(labels)
            x_array[start:stop] = np.asarray(xs, dtype=np.int32)
            y_array[start:stop] = np.asarray(ys, dtype=np.int32)
        for index in sorted({0, n // 2, n - 1}):
            with np.load(files[index], allow_pickle=False) as item:
                if not np.array_equal(np.asarray(item["image"], dtype=np.float32), image_array[index], equal_nan=True):
                    raise RuntimeError(f"Lossless image check failed: {files[index]}")
                if not np.array_equal(np.asarray(item["label"], dtype=np.uint8), label_array[index]):
                    raise RuntimeError(f"Lossless label check failed: {files[index]}")
        root.attrs["complete"] = True
        # Windows refuses to rename a directory while Zarr/numcodecs still
        # holds a file handle.  Close the DirectoryStore explicitly before the
        # atomic move; relying on garbage collection is intermittent.
        store = root.store
        del image_array, label_array, x_array, y_array, group, root
        gc.collect()
        close = getattr(store, "close", None)
        if close is not None:
            close()
        del store
        print(f"[Packed] ID{sid}: {n} {row['split']} patches", flush=True)
        return records
    except Exception:
        print(f"[Preserved incomplete store] {destination}", flush=True)
        raise


def rebuild_indices(output: Path, source: Path, manifest: Path, rows: list[dict], new_records: list[dict], level: int) -> None:
    index_rows, patch_rows = [], list(new_records)
    for row in rows:
        path = output / f"ID{row['sample_id']:02d}.zarr"
        root = zarr.open_group(str(path), mode="r")
        if not bool(root.attrs.get("complete", False)):
            raise RuntimeError(f"Incomplete store: {path}")
        index_rows.append({**row, "store": path.name, "image_dtype": "float32", "label_dtype": "uint8"})
        if not any(r["sample_id"] == row["sample_id"] for r in patch_rows):
            files = source_files(source, row)
            for i, file in enumerate(files):
                with np.load(file, allow_pickle=False) as item:
                    patch_rows.append(
                        dict(sample_id=row["sample_id"], split=row["split"], class_name=row["class_name"], store_index=i,
                             source_file=file.name, x=_scalar(item, "x", -1), y=_scalar(item, "y", -1))
                    )
    for filename, records in (("index.csv", index_rows), ("patch_index.csv", patch_rows)):
        temporary = output / f".{filename}.tmp"
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        os.replace(temporary, output / filename)
    listing = "\n".join(sorted(str(p.relative_to(source)) for row in rows for p in source_files(source, row)))
    provenance = {
        "format": "AEF_FIXED_SPLIT_ZARR_V1",
        "source": str(source),
        "manifest": str(manifest),
        "source_listing_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
        "compressor": "zstd",
        "compression_level": level,
        "chunk": [1, 64, 128, 128],
        "train_samples": sum(r["split"] == "train" for r in rows),
        "val_samples": sum(r["split"] == "val" for r in rows),
        "train_patches": sum(r["patches"] for r in rows if r["split"] == "train"),
        "val_patches": sum(r["patches"] for r in rows if r["split"] == "val"),
        "design": "Exact fixed fold0 partition from 01-processed_cv; one lossless Zarr store per sample; no resampling.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Losslessly pack the original fixed fold0 NPZ split into per-sample Zarr stores.")
    parser.add_argument("--source", type=Path, default=Path(cfg.source_npz))
    parser.add_argument("--output", type=Path, default=Path(cfg.data_path))
    parser.add_argument("--manifest", type=Path, default=Path(cfg.manifest_path))
    parser.add_argument("--compression-level", type=int, default=7)
    parser.add_argument("--write-batch", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, manifest = args.source.resolve(), args.manifest.resolve()
    rows = read_manifest(manifest)
    if {r["sample_id"] for r in rows} & {4, 5}:
        raise ValueError("ID4/ID5 must not enter this fixed-split project")
    for row in rows:
        files = source_files(source, row)
        print(f"ID{row['sample_id']:02d} {row['split']:5s} {row['class_name']:10s} {len(files):4d}")
    if args.dry_run:
        print("Dry run complete; no files written.")
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    new_records = []
    for row in rows:
        new_records.extend(build_sample(output, source, row, codec(args.compression_level), args.write_batch))
    rebuild_indices(output, source, manifest, rows, new_records, args.compression_level)
    print(f"Ready: {output}")


if __name__ == "__main__":
    main()
