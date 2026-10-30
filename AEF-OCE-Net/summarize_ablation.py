from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .configuration import Config


RUNS = {
    "Full C3": ("C3_hierarchical", "scratch_v1"),
    "minus Adapter": ("C3_minus_adapter", "fixedsplit_seed2026"),
    "minus OCE": ("C3_minus_oce", "fixedsplit_seed2026"),
}


def main():
    parser=argparse.ArgumentParser(description="Summarize the full model and two strict single-factor ablations.")
    parser.add_argument("--output",type=Path,default=None); args=parser.parse_args(); cfg=Config()
    output=args.output.resolve() if args.output else cfg.output_dir/'ablation_summary'; output.mkdir(parents=True,exist_ok=True)
    metric_rows=[]; sample_parts=[]
    for label,(variant,run_name) in RUNS.items():
        run=cfg.output_dir/variant/run_name
        done=run/'DONE.json'; metrics=run/'best_metrics.json'; samples=run/'best_per_sample_metrics.csv'
        if not (done.exists() and metrics.exists() and samples.exists()): raise FileNotFoundError(f"Incomplete run: {run}")
        values=json.loads(metrics.read_text(encoding='utf-8')); status=json.loads(done.read_text(encoding='utf-8'))
        metric_rows.append({'model':label,'variant':variant,'run_name':run_name,'best_epoch':status['best_epoch'],**values})
        frame=pd.read_csv(samples); frame['model']=label
        frame['target_recall']=np.select([frame.class_name.eq('background'),frame.class_name.eq('mango'),frame.class_name.eq('lychee')],[frame.background_recall,frame.mango_recall,frame.lychee_recall])
        sample_parts.append(frame)
    metrics=pd.DataFrame(metric_rows); samples=pd.concat(sample_parts,ignore_index=True)
    metrics.to_csv(output/'Table_ablation_metrics.csv',index=False,encoding='utf-8-sig')
    full=metrics.set_index('model').loc['Full C3']; deltas=[]
    for name in ('minus Adapter','minus OCE'):
        row=metrics.set_index('model').loc[name]
        deltas.append({'ablation':name,**{f'delta_full_minus_ablation_{m}':full[m]-row[m] for m in ('miou','mango_f1','lychee_f1','orchard_to_background','mango_to_lychee','lychee_to_mango')}})
    pd.DataFrame(deltas).to_csv(output/'Table_ablation_deltas.csv',index=False,encoding='utf-8-sig')
    pivot=samples.pivot(index=['sample_id','county','class_name'],columns='model',values='target_recall').reset_index()
    pivot['gain_vs_minus_adapter']=pivot['Full C3']-pivot['minus Adapter']; pivot['gain_vs_minus_oce']=pivot['Full C3']-pivot['minus OCE']
    pivot.to_csv(output/'Table_per_sample_target_recall_gains.csv',index=False,encoding='utf-8-sig')

    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.08,3.2),constrained_layout=True)
    names=metrics.model.tolist(); x=np.arange(len(names)); width=.22
    for offset,(metric,label,color) in zip((-width,0,width),(('miou','mIoU','#365F91'),('mango_f1','Mango F1','#D97706'),('lychee_f1','Lychee F1','#16837A'))):
        axes[0].bar(x+offset,metrics[metric],width,label=label,color=color)
    axes[0].set_xticks(x,names,rotation=15); axes[0].set_ylim(.7,1); axes[0].set_ylabel('Fixed-split score'); axes[0].legend(frameon=False); axes[0].set_title('a  Overall ablation performance',loc='left',fontweight='bold')
    y=np.arange(len(pivot)); axes[1].axvline(0,color='#777777',lw=.8)
    axes[1].scatter(pivot.gain_vs_minus_adapter,y,label='Adapter contribution',color='#6F4E9C',s=24)
    axes[1].scatter(pivot.gain_vs_minus_oce,y,label='OCE contribution',color='#16837A',s=24,marker='s')
    axes[1].set_yticks(y,[f"ID{sid}" for sid in pivot.sample_id]); axes[1].set_xlabel('Full minus ablation: target-class recall'); axes[1].legend(frameon=False); axes[1].set_title('b  Per-sample module contribution',loc='left',fontweight='bold')
    fig.savefig(output/'Fig_ablation_summary.pdf',bbox_inches='tight'); fig.savefig(output/'Fig_ablation_summary.png',dpi=500,bbox_inches='tight'); plt.close(fig)
    print(f"Ablation summary ready: {output}")


if __name__=='__main__': main()

