"""Run v0.4 tabular models, respecting use_train=1 only for training.
Usage: python train_tabular_v3.py --model lgbm|mlp|xgb --data data/data_v0.4/train_v0.4.parquet
XGBoost requires NVIDIA CUDA by default; optionally --device cpu for local smoke test.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from oibc_v3_common import read_v3, train_mask, standardize, categoricals, result_files

IDS={'lgbm':'exp01_lgbm_v3','mlp':'exp02_mlp_v3','xgb':'exp03_xgb_gpu_v3'}

def run_lgbm(train, dev, cats, nums, args):
    import lightgbm as lgb
    t0=time.time()
    xtr=train[cats+nums].copy(); xdev=dev[cats+nums].copy()
    for c in cats:
        vocab=pd.Index(train[c].dropna().unique()).tolist()
        xtr[c]=pd.Categorical(xtr[c],categories=vocab)
        xdev[c]=pd.Categorical(xdev[c],categories=vocab)
    for c in nums:
        xtr[c]=pd.to_numeric(xtr[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
        xdev[c]=pd.to_numeric(xdev[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
    y=train.cf_target.to_numpy(dtype=np.float32,copy=True)
    model=lgb.LGBMRegressor(objective='regression_l1',n_estimators=650,learning_rate=.05,
                             num_leaves=31,n_jobs=4,random_state=42,verbosity=-1)
    daylight=(dev.is_daylight.to_numpy()==1) if 'is_daylight' in dev else (dev.elev.to_numpy()>0)
    model.fit(xtr,y,eval_set=[(xdev.loc[daylight],dev.cf_target.to_numpy(dtype=np.float32)[daylight])],
              eval_metric='l1',callbacks=[lgb.early_stopping(70,verbose=True),lgb.log_evaluation(50)])
    cf=model.predict(xdev,num_iteration=model.best_iteration_)
    pred=np.maximum(cf,0)*dev.capacity_kw.to_numpy(dtype=float)
    model.booster_.save_model(str(Path(args.models)/f'{IDS["lgbm"]}.txt'))
    return pred, {'best_iteration':model.best_iteration_,'train_seconds':round(time.time()-t0,1)}

def run_xgb(train,dev,cats,nums,args):
    import xgboost as xgb
    if args.device=='cuda':
        import subprocess
        subprocess.run(['nvidia-smi'],check=True,stdout=subprocess.DEVNULL)
        if str(xgb.build_info().get('USE_CUDA','True')).lower() in ('false','0','off'):
            raise RuntimeError('CUDA 미지원 XGBoost입니다. Colab GPU에서 실행하세요.')
    xtr=train[cats+nums].copy(); xdev=dev[cats+nums].copy()
    for c in cats:
        vocab=pd.Index(train[c].dropna().unique()).tolist()
        xtr[c]=pd.Categorical(xtr[c],categories=vocab)
        xdev[c]=pd.Categorical(xdev[c],categories=vocab)
    for c in nums:
        xtr[c]=pd.to_numeric(xtr[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
        xdev[c]=pd.to_numeric(xdev[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
    t0=time.time()
    model=xgb.XGBRegressor(objective='reg:absoluteerror',eval_metric='mae',n_estimators=900,
              learning_rate=.05,max_depth=6,min_child_weight=15,subsample=.85,
              colsample_bytree=.85,max_bin=128,reg_lambda=2,tree_method='hist',device=args.device,
              enable_categorical=True,early_stopping_rounds=70,random_state=42,n_jobs=4)
    daylight=(dev.is_daylight.to_numpy()==1) if 'is_daylight' in dev else (dev.elev.to_numpy()>0)
    model.fit(xtr,train.cf_target.to_numpy(dtype=np.float32,copy=True),
              eval_set=[(xdev.loc[daylight],dev.cf_target.to_numpy(dtype=np.float32,copy=True)[daylight])],verbose=50)
    config=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
    if args.device=='cuda' and not str(config).startswith('cuda'): raise RuntimeError('GPU 학습이 실행되지 않았습니다.')
    cf=model.predict(xdev)
    pred=np.maximum(cf,0)*dev.capacity_kw.to_numpy(dtype=float)
    model.save_model(str(Path(args.models)/f'{IDS["xgb"]}.json'))
    return pred, {'best_iteration':int(model.best_iteration), 'device':config,'train_seconds':round(time.time()-t0,1)}


def run_mlp(train,dev,cats,nums,args):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('MLP 학습 장치:',device)
    xt,xv,stats=standardize(train,dev,nums)
    ct,cv,vocab=categoricals(train,dev,cats)
    em_dims=[16 if c=='site_id' else 8 for c in cats]
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb=nn.ModuleList([nn.Embedding(len(vocab[c])+1,dim,padding_idx=0)
                for c,dim in zip(cats,em_dims)])
            n=len(nums)+sum(em_dims)
            self.net=nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.Dropout(.1),
                  nn.Linear(256,128),nn.ReLU(),nn.Dropout(.1),
                  nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        def forward(self,num,cat):
            x=torch.cat([num]+[emb(cat[:,i]) for i,emb in enumerate(self.emb)],dim=-1)
            return self.net(x).squeeze(-1)
    model=Model().to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(ct),
                  torch.from_numpy(train.cf_target.to_numpy(dtype=np.float32,copy=True))),
                  batch_size=4096,shuffle=True,num_workers=0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    loss_fn=nn.SmoothL1Loss(beta=.05)
    def infer():
        model.eval(); parts=[]
        with torch.no_grad():
            for i in range(0,len(xv),8192):
                a=torch.from_numpy(xv[i:i+8192]).to(device)
                b=torch.from_numpy(cv[i:i+8192]).to(device)
                parts.append(model(a,b).float().cpu().numpy())
        return np.concatenate(parts).astype(np.float64)
    best=float('inf'); wait=0; epoch_best=0; best_state=None;t0=time.time()
    for epoch in range(1,26):
        model.train(); tot=0;n=0
        for a,b,y in loader:
            a,b,y=a.to(device),b.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(a,b),y);loss.backward();optimizer.step()
            tot+=loss.item()*len(y);n+=len(y)
        cf=infer();pred=np.clip(cf,0,None)*dev.capacity_kw.to_numpy(dtype=float)
        if 'is_daylight' in dev:
            pred=np.where(dev.is_daylight.to_numpy()==1,pred,0.)
        mae=float(np.mean(np.abs(pred-dev.energy_kwh.to_numpy(dtype=float))))
        print(f'epoch {epoch:02d}: train_loss={tot/n:.5f}, dev_MAE={mae:.4f}kWh',flush=True)
        if mae<best-1e-5:
            best=mae;epoch_best=epoch;wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=5: print('Early stopping');break
    model.load_state_dict(best_state); cf=infer()
    pred=np.clip(cf,0,None)*dev.capacity_kw.to_numpy(dtype=float)
    torch.save({'state_dict':best_state,'numeric_features':nums,'categorical_features':cats,
        'vocab':vocab,'stats':stats,'best_epoch':epoch_best,'data_version':'v0.4'},
        Path(args.models)/f'{IDS["mlp"]}.pt')
    return pred, {'best_epoch':epoch_best,'device':device,'train_seconds':round(time.time()-t0,1)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=IDS,required=True)
    ap.add_argument('--data',default=None,help='v0.4 train parquet')
    ap.add_argument('--models',default='models');ap.add_argument('--outputs',default='outputs')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='XGBoost GPU setting')
    a=ap.parse_args()
    Path(a.models).mkdir(parents=True,exist_ok=True)
    alltrain,dev,_,cats,nums,_=read_v3(a.data)
    train=alltrain.loc[train_mask(alltrain)].copy()
    if train.cf_target.isna().any(): raise ValueError('학습 cf_target에 NaN이 있습니다.')
    print(f'학습 대상(use_train=1): {len(train):,}행 / Dev 평가: {len(dev):,}행 (전체)')
    f={'lgbm':run_lgbm,'mlp':run_mlp,'xgb':run_xgb}[a.model]
    pred,info=f(train,dev,cats,nums,a)
    result_files(IDS[a.model],a.outputs,a.models,dev,pred,
          {**info,'train_rows':len(train),'categorical_features':cats,'numeric_features':nums})

if __name__=='__main__': main()
