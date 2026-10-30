from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import zarr

from .configuration import Config


def main():
    parser = argparse.ArgumentParser(description="Read the same fixed-split patches from NPZ and Zarr.")
    parser.add_argument("--patches", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(); cfg = Config()
    index = pd.read_csv(cfg.data_dir / "patch_index.csv").sample(n=args.patches, random_state=cfg.seed).reset_index(drop=True)
    groups = {}; records = []
    for repeat in range(args.repeats):
        for backend in ("npz", "zarr"):
            started=time.perf_counter(); checksum=0.0
            for row in index.itertuples(index=False):
                if backend=="npz":
                    path=f"{cfg.source_npz}/fold0/{row.split}/{row.class_name}/{row.source_file}"
                    with np.load(path,allow_pickle=False) as item:
                        image=np.asarray(item['image']); label=np.asarray(item['label'])
                else:
                    sid=int(row.sample_id)
                    if sid not in groups: groups[sid]=zarr.open_group(str(cfg.data_dir/f"ID{sid:02d}.zarr"),mode='r')['patches']
                    image=np.asarray(groups[sid]['image'][int(row.store_index)]); label=np.asarray(groups[sid]['label'][int(row.store_index)])
                checksum += float(np.nan_to_num(image[0,0,0], nan=0.0, posinf=0.0, neginf=0.0)) + float(label[0,0])
            elapsed=time.perf_counter()-started
            records.append({"repeat":repeat,"backend":backend,"patches":args.patches,"seconds":elapsed,"patches_per_second":args.patches/elapsed,"checksum":checksum})
    output=cfg.output_dir.parent; output.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(records); table.to_csv(output/'storage_read_benchmark.csv',index=False,encoding='utf-8-sig')
    summary=table.groupby('backend').patches_per_second.agg(['mean','std','min','max']).reset_index().to_dict('records')
    (output/'storage_read_benchmark.json').write_text(json.dumps({"method":"identical randomly selected patches; full image+label decode","raw":records,"summary":summary},indent=2),encoding='utf-8')
    print(table.to_string(index=False)); print(summary)


if __name__=='__main__': main()
