"""Article 2 - 2,000-replicate bootstrap intervals for fixed logistic-regression test predictions."""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_recall_curve, f1_score, matthews_corrcoef,
    average_precision_score, balanced_accuracy_score, precision_score, recall_score)
EXCLUDE={'ts','Label','gap_min','segment','y10','y30','y60'}
def choose_threshold(y,p):
    pr,rc,th=precision_recall_curve(y,p)
    f=2*pr*rc/np.maximum(pr+rc,1e-12)
    i=int(np.nanargmax(f[:-1]))
    return float(th[i])
def main(inp,out,nboot=2000,seed=20261016):
    d=pd.read_csv(inp,parse_dates=['ts']); F=[c for c in d.columns if c not in EXCLUDE]
    tr=d[d.ts<'2026-05-19']; va=d[(d.ts>='2026-05-19')&(d.ts<'2026-05-20')]; te=d[d.ts>='2026-05-20']
    rng=np.random.default_rng(seed); R={}
    for H in (10,30,60):
        ytr=tr[f'y{H}'].to_numpy(); yv=va[f'y{H}'].to_numpy(); yt=te[f'y{H}'].to_numpy()
        mdl=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced',random_state=42))
        mdl.fit(tr[F],ytr)
        thr=choose_threshold(yv,mdl.predict_proba(va[F])[:,1])
        p=mdl.predict_proba(te[F])[:,1]; z=(p>=thr).astype(int)
        vals={k:[] for k in ('f1','mcc','prauc','bacc','precision','recall')}
        for _ in range(nboot):
            idx=rng.integers(0,len(yt),len(yt)); yy=yt[idx]; pp=p[idx]; zz=z[idx]
            if len(np.unique(yy))<2: continue
            vals['f1'].append(f1_score(yy,zz,zero_division=0))
            vals['mcc'].append(matthews_corrcoef(yy,zz))
            vals['prauc'].append(average_precision_score(yy,pp))
            vals['bacc'].append(balanced_accuracy_score(yy,zz))
            vals['precision'].append(precision_score(yy,zz,zero_division=0))
            vals['recall'].append(recall_score(yy,zz,zero_division=0))
        R[str(H)]={'threshold':thr,'ci95':{k:[float(np.quantile(v,.025)),float(np.quantile(v,.975))] for k,v in vals.items()}}
    json.dump(R,open(out,'w'),indent=2)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('features_csv'); ap.add_argument('--out',default='bootstrap_results.json')
    ap.add_argument('--nboot',type=int,default=2000)
    a=ap.parse_args(); main(a.features_csv,a.out,a.nboot)
