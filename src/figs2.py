"""Article 2 - generate paper figures from the saved JSON and feature table."""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(features,results,outdir):
    import os; os.makedirs(outdir,exist_ok=True)
    d=pd.read_csv(features,parse_dates=['ts']); R=json.load(open(results))
    # Fig 1: positives by split and horizon
    labels=['Train','Validation','Test']; x=np.arange(3); w=.24
    fig,ax=plt.subplots(figsize=(5.4,3.1))
    masks=[d.ts<'2026-05-19',(d.ts>='2026-05-19')&(d.ts<'2026-05-20'),d.ts>='2026-05-20']
    for j,H in enumerate((10,30,60)):
        vals=[int(d.loc[m,f'y{H}'].sum()) for m in masks]; ax.bar(x+(j-1)*w,vals,width=w,label=f'{H} min')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Positive early-warning minutes'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir+'/fig1_target_counts.png',dpi=300); plt.close(fig)
    # Fig 2: model F1 by horizon
    models=['Logistic','Random Forest','Extra Trees','LightGBM','XGBoost','CatBoost']; x=np.arange(len(models)); w=.24
    fig,ax=plt.subplots(figsize=(6.3,3.2))
    for j,H in enumerate((10,30,60)):
        vals=[100*R['horizons'][str(H)][m]['f1'] for m in models]; ax.bar(x+(j-1)*w,vals,width=w,label=f'{H} min')
    ax.set_xticks(x); ax.set_xticklabels(models,rotation=25,ha='right'); ax.set_ylabel('Test F1-score (%)'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir+'/fig2_model_f1.png',dpi=300); plt.close(fig)
    # Fig 3: current-only vs temporal ablation
    fig,ax=plt.subplots(figsize=(5.2,3.0)); x=np.arange(3); w=.32
    for j,n in enumerate(['Current-only','Current+temporal']):
        vals=[100*R['ablation'][str(H)][n]['f1'] for H in (10,30,60)]; ax.bar(x+(j-.5)*w,vals,width=w,label=n)
    ax.set_xticks(x); ax.set_xticklabels(['10 min','30 min','60 min']); ax.set_ylabel('Test F1-score (%)'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir+'/fig3_temporal_ablation.png',dpi=300); plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('features_csv'); ap.add_argument('results_json'); ap.add_argument('--outdir',default='figures')
    a=ap.parse_args(); main(a.features_csv,a.results_json,a.outdir)
