from __future__ import annotations

import json

import numpy as np
import pandas as pd
import zarr

from .configuration import Config


def main():
    cfg=Config(); root=cfg.data_dir; index=pd.read_csv(root/'index.csv')
    allowed={'background':{0,255},'mango':{0,1,255},'lychee':{0,2,255}}
    total=valid=ignored=foreground=mismatch=0; records=[]
    for row in index.itertuples(index=False):
        group=zarr.open_group(str(root/row.store),mode='r')['patches']
        labels=np.asarray(group['label']); images=np.asarray(group['image'])
        unique=set(np.unique(labels).tolist())
        if not unique<=allowed[row.class_name]: raise ValueError(f'ID{row.sample_id}: invalid labels {unique}')
        aef_valid=np.all(np.isfinite(images),axis=1)&~np.all(images<=-127.5,axis=1)
        label_valid=labels!=255; bad=label_valid&~aef_valid
        record={'sample_id':int(row.sample_id),'split':row.split,'class_name':row.class_name,'pixels':int(labels.size),'valid_pixels':int(label_valid.sum()),'foreground_pixels':int(((labels>0)&(labels<255)).sum()),'valid_label_on_invalid_aef':int(bad.sum())}
        records.append(record); total+=record['pixels']; valid+=record['valid_pixels']; foreground+=record['foreground_pixels']; mismatch+=record['valid_label_on_invalid_aef']; ignored+=int((labels==255).sum())
    report={'pixels':total,'valid_label_pixels':valid,'ignored_pixels':ignored,'foreground_pixels':foreground,'valid_label_on_invalid_aef':mismatch,'mismatch_fraction':mismatch/max(valid,1),'worst_samples':sorted(records,key=lambda r:r['valid_label_on_invalid_aef'],reverse=True)[:8]}
    output=cfg.output_dir.parent; output.mkdir(parents=True,exist_ok=True); pd.DataFrame(records).to_csv(output/'label_audit_by_sample.csv',index=False,encoding='utf-8-sig'); (output/'label_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()

