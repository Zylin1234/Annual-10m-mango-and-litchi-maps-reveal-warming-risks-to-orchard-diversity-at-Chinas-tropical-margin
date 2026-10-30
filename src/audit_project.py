from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import zarr

from .configuration import Config
from .data import FixedSplitZarrDataset, read_manifest
from .losses import HierarchicalSegmentationLoss
from .model import VARIANTS, build_model


def main():
    cfg = Config(); cfg.validate(); manifest = read_manifest(cfg.manifest_file)
    train_ids = {r["sample_id"] for r in manifest if r["split"] == "train"}; val_ids = {r["sample_id"] for r in manifest if r["split"] == "val"}
    assert not train_ids & val_ids and not ({4,5} & (train_ids|val_ids))
    assert len(train_ids)==20 and len(val_ids)==6
    train = FixedSplitZarrDataset(cfg.data_dir, manifest, "train", augment=False); val = FixedSplitZarrDataset(cfg.data_dir, manifest, "val", augment=False)
    assert len(train)==5752 and len(val)==409
    x, y, meta = train[0]; assert x.shape==(65,128,128) and y.shape==(128,128)
    # One source-vs-store equality check per sample complements the three
    # within-sample checks performed while writing each Zarr store.
    patch_index = pd.read_csv(cfg.data_dir / "patch_index.csv")
    source_root = cfg.source_npz
    lossless_checks = 0
    for row in manifest:
        record = patch_index[patch_index.sample_id == row["sample_id"]].sort_values("store_index").iloc[0]
        source = f"{source_root}/fold0/{row['split']}/{row['class_name']}/{record.source_file}"
        with np.load(source, allow_pickle=False) as item:
            group = zarr.open_group(str(cfg.data_dir / f"ID{row['sample_id']:02d}.zarr"), mode="r")["patches"]
            assert np.array_equal(np.asarray(item["image"], dtype=np.float32), group["image"][int(record.store_index)], equal_nan=True)
            assert np.array_equal(np.asarray(item["label"], dtype=np.uint8), group["label"][int(record.store_index)])
        lossless_checks += 1
    target = torch.tensor([[[0]*16]*16, [[1]*16]*16, [[2]*16]*16], dtype=torch.long)
    synthetic = torch.randn(3,65,16,16); synthetic[:,64]=1
    checks = {}
    for variant in VARIANTS:
        local = Config(variant=variant); model = build_model(local); outputs = model(synthetic)
        assert outputs["logits"].shape==(3,3,16,16) and torch.isfinite(outputs["logits"]).all()
        criterion = HierarchicalSegmentationLoss(local); loss,_=criterion(outputs,target,epoch=12); loss.backward(); assert torch.isfinite(loss)
        checks[variant]={"parameters":sum(p.numel() for p in model.parameters()),"loss":float(loss)}
        if variant=="C3_hierarchical":
            total=torch.softmax(outputs["logits"],1).sum(1); assert torch.allclose(total,torch.ones_like(total),atol=1e-6)
            assert torch.allclose(outputs["context_weights"].sum(1),torch.ones_like(total),atol=1e-6)
            for prefix in ("adapter", "oce", "orchard_head", "species_head"):
                grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)
                assert grad>0, f"No gradient reached {prefix}"
    # One real-data optimizer step covering all three semantic groups.
    selected = [train.sample_classes.index(name) for name in ("background", "mango", "lychee")]
    actual = [train[i] for i in selected]
    actual_x = torch.stack([item[0][:, 32:96, 32:96] for item in actual])
    actual_y = torch.stack([item[1][32:96, 32:96] for item in actual])
    actual_cfg = Config(variant="C3_hierarchical"); actual_model = build_model(actual_cfg)
    optimizer = torch.optim.AdamW(actual_model.parameters(), lr=1e-4)
    before = actual_model.orchard_head[-1].weight.detach().clone()
    actual_outputs = actual_model(actual_x); actual_loss, _ = HierarchicalSegmentationLoss(actual_cfg)(actual_outputs, actual_y, 12)
    optimizer.zero_grad(); actual_loss.backward(); optimizer.step()
    assert torch.isfinite(actual_loss) and not torch.equal(before, actual_model.orchard_head[-1].weight.detach())
    report={"train_ids":sorted(train_ids),"val_ids":sorted(val_ids),"train_patches":len(train),"val_patches":len(val),"class_patches":Counter(train.sample_classes),"lossless_source_store_samples_checked":lossless_checks,"real_data_optimizer_step_loss":float(actual_loss),"variants":checks,"input_valid_source":"AEF finite/nodata mask, never label coverage","status":"PASS"}
    path=cfg.output_dir.parent / "project_audit.json"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=dict),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2,default=dict))


if __name__ == "__main__": main()
