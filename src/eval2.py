"""Article 2 - chronological training/validation/test evaluation for 10/30/60 min horizons."""
import argparse, json, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
EXCLUDE={'ts','Label','gap_min','segment','y10','y30','y60'}
CURRENT=['Soil Percent','Soil Raw','Suhu Tanah','Suhu Udara','Humidity']
def choose_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2*pr*rc/np.maximum(pr+rc,1e-12)
    i = int(np.nanargmax(f[:-1]))
    return float(th[i]), float(f[i])
def score(y,p,t):
    z=(p>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,z),balanced_accuracy=balanced_accuracy_score(y,z),
        precision=precision_score(y,z,zero_division=0),recall=recall_score(y,z,zero_division=0),
        f1=f1_score(y,z,zero_division=0),mcc=matthews_corrcoef(y,z),
        pr_auc=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=float(t))
def main(inp,out):
    d=pd.read_csv(inp,parse_dates=['ts']); F=[c for c in d.columns if c not in EXCLUDE]
    tr=d[d.ts<'2026-05-19']; va=d[(d.ts>='2026-05-19')&(d.ts<'2026-05-20')]; te=d[d.ts>='2026-05-20']
    R={'meta':{'n_train':len(tr),'n_validation':len(va),'n_test':len(te),'n_features':len(F)},'horizons':{},'ablation':{}}
    for H in (10,30,60):
        ytr=tr[f'y{H}'].to_numpy(); yv=va[f'y{H}'].to_numpy(); yt=te[f'y{H}'].to_numpy()
        ratio=(len(ytr)-ytr.sum())/max(1,ytr.sum())
        models={
            'Logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced',random_state=42)),
            'Random Forest':RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=42),
            'Extra Trees':ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',n_jobs=-1,random_state=42),
            'LightGBM':lgb.LGBMClassifier(n_estimators=250,learning_rate=.03,num_leaves=15,scale_pos_weight=ratio,random_state=42,verbosity=-1),
            'XGBoost':xgb.XGBClassifier(n_estimators=250,learning_rate=.03,max_depth=4,subsample=.9,colsample_bytree=.9,scale_pos_weight=ratio,eval_metric='logloss',random_state=42,n_jobs=-1),
            'CatBoost':CatBoostClassifier(iterations=250,learning_rate=.03,depth=6,auto_class_weights='Balanced',verbose=False,random_seed=42,thread_count=-1),
        }
        R['horizons'][str(H)]={}
        for name,m in models.items():
            t0=time.time(); m.fit(tr[F],ytr); fit=time.time()-t0
            pv=m.predict_proba(va[F])[:,1]; thr,val_f1=choose_threshold(yv,pv)
            pt=m.predict_proba(te[F])[:,1]; met=score(yt,pt,thr); met['validation_f1']=val_f1; met['fit_s']=fit
            R['horizons'][str(H)][name]=met
            print(H,name,'F1',round(met['f1'],4),'MCC',round(met['mcc'],4),'PR-AUC',round(met['pr_auc'],4))
        # Ablation: same logistic estimator, current readings only versus all causal temporal features.
        R['ablation'][str(H)]={}
        for name,cols in [('Current-only',CURRENT),('Current+temporal',F)]:
            m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced',random_state=42))
            m.fit(tr[cols],ytr); pv=m.predict_proba(va[cols])[:,1]; thr,vf=choose_threshold(yv,pv); pt=m.predict_proba(te[cols])[:,1]
            met=score(yt,pt,thr); met['validation_f1']=vf; R['ablation'][str(H)][name]=met
    json.dump(R,open(out,'w'),indent=2)
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('features_csv'); ap.add_argument('--out',default='forecast_results.json')
    a=ap.parse_args(); main(a.features_csv,a.out)
