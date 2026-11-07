"""OIBC v0.4: baseline-style LightGBM round10/17 mean DEV adaptation.
NOT identical to competition baseline.py: operates on v0.4 prepared features,
uses use_train=1, has a train/dev split, and may omit absent forecast columns.
"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
from oibc_v3_common import read_v3,train_mask,result_files

WA=['wa_cloud','wa_temp','wa_humidity','wa_rain','wa_wind_speed','wa_ground_press']
WB=['wb_cloud','wb_temp','wb_rel_hum','wb_uv_idx','wb_precip_prob',
    'wb_total_liq','wb_vis','wb_wind_speed','wb_dew_point']
NAME='exp00_official_baseline_v3_dev_adapted'

def main():
    import lightgbm as lgb
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=None);p.add_argument('--models',default='models');p.add_argument('--outputs',default='outputs')
    a=p.parse_args();Path(a.models).mkdir(exist_ok=True,parents=True)
    trall,dev,_,cats,nums,_=read_v3(a.data)
    tr=trall.loc[train_mask(trall)].copy()
    # Same baseline idea: 2 independently trained models, each sees a single
    # forecast round's weather plus a small static/time feature group.
    static=[c for c in ['elev','azim','cosz','doy_sin','doy_cos','hour','capacity_kw','lat','lon']
            if c in tr.columns or c=='hour']
    tr['hour']=tr.ts.dt.hour.astype('float32');dev['hour']=dev.ts.dt.hour.astype('float32')
    roundpred=[];counts=[];t0=time.time()
    for r in (10,17):
        columns=[]
        for c in WA+WB:
            candidates=[f'{c}_r{r}']
            if c=='wb_wind_speed': candidates.append(f'wb_wind_speed_ms_r{r}')
            for cand in candidates:
                if cand in tr.columns:
                    columns.append(cand);break
        feats=columns+static
        if len(columns)<8:
            raise ValueError(f'라운드 {r}: 베이스라인 계열 기상변수가 {len(columns)}개뿐입니다. 원본 v0.4 스키마 확인 필요.')
        print(f'[EXP00] r{r}: 특징 {len(feats)}개: {feats}')
        xtrain=tr[feats].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
        xdev=dev[feats].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
        model=lgb.LGBMRegressor(n_estimators=600,learning_rate=.06,num_leaves=127,
                  min_child_samples=200,subsample=.8,subsample_freq=1,
                  colsample_bytree=.8,verbose=-1,n_jobs=4,random_state=42)
        model.fit(xtrain,tr.cf_target.to_numpy(dtype=np.float32,copy=True))
        model.booster_.save_model(str(Path(a.models)/f'exp00_official_baseline_v3_round{r}.txt'))
        roundpred.append(model.predict(xdev)); counts.append(len(feats))
    cf=(roundpred[0]+roundpred[1])/2
    pred=np.clip(cf,0,None)*dev.capacity_kw.to_numpy(dtype=float)
    result_files(NAME,a.outputs,a.models,dev,pred,{'train_rows':len(tr),
        'train_seconds':round(time.time()-t0,1),'forecast_rounds':[10,17],
        'feature_counts_per_round':counts,
        'warning':'baseline STYLE Dev adaptation, not an unmodified official baseline.py run'})

if __name__=='__main__':main()
