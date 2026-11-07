"""OIBC v0.4 24-hour sequence models: Transformer / bidirectional LSTM.
A right-closed delivery day consists of 01:00..23:00 then next 00:00 KST.
Training loss is masked to use_train=1; Dev predictions cover EVERY row.
DO NOT use May holdout for model choice; no pretrained weights or external data.
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from oibc_v3_common import read_v3, train_mask, standardize, categoricals, result_files

EXPS={'transformer':'exp05_transformer_v3','lstm':'exp06_lstm_v3'}

def prepare_sequence(frame, xnum, xcat, *, train):
    # Preprocessed arrays must align to original df row order.
    extra=[c for c in ('elev','is_daylight') if c in frame.columns]
    df=frame[['site_id','ts','capacity_kw','energy_kwh','cf_target','use_train']+extra].copy()
    df['delivery_day']=(df.ts-pd.Timedelta(hours=1)).dt.date
    # Use a stable index permutation, then apply exactly the same to inputs.
    order=df.sort_values(['site_id','delivery_day','ts'],kind='stable').index
    rowpos=pd.Series(np.arange(len(frame)),index=frame.index)
    pos=rowpos.loc[order].to_numpy(dtype=np.int64)
    df=df.loc[order].reset_index(drop=True)
    xnum=xnum[pos]; xcat=xcat[pos]
    grp=pd.MultiIndex.from_frame(df[['site_id','delivery_day']])
    code,unique=pd.factorize(grp,sort=False)
    count=np.bincount(code)
    ok=count[code]==24
    if not train and not bool(ok.all()):
        raise ValueError(f'Dev는 날짜·발전소별 24시간이 모두 필요합니다. 불완전 행 {(~ok).sum()}개')
    df=df.loc[ok].reset_index(drop=True)
    xnum=xnum[ok];xcat=xcat[ok]
    if len(df)%24: raise RuntimeError('24시간 reshape 불가')
    ts_ns=df.ts.array.asi8.reshape(-1,24)
    if not np.all(np.diff(ts_ns,axis=1)==3600_000_000_000):
        raise ValueError('24시간 예측 단위의 시간축이 연결되지 않았습니다')
    yy=df.cf_target.to_numpy(dtype=np.float32,copy=True).reshape(-1,24)
    mask=train_mask(df).reshape(-1,24) if train else np.ones_like(yy,dtype=bool)
    if train:
        keep=mask.any(axis=1)
        xnum=xnum.reshape(-1,24,xnum.shape[-1])[keep]
        xcat=xcat.reshape(-1,24,xcat.shape[-1])[keep]
        yy=yy[keep];mask=mask[keep]
        if not np.isfinite(yy[mask]).all(): raise ValueError('학습 대상 cf_target NaN')
        yy=np.where(mask,yy,0.)
        return xnum,xcat,yy,mask,None
    return (xnum.reshape(-1,24,xnum.shape[-1]),xcat.reshape(-1,24,xcat.shape[-1]),
            yy,mask,df)

def main():
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=EXPS,required=True)
    ap.add_argument('--data',default=None)
    ap.add_argument('--models',default='models');ap.add_argument('--outputs',default='outputs')
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--batch-size',type=int,default=128)
    ap.add_argument('--patience',type=int,default=5)
    args=ap.parse_args()
    if args.batch_size<1: raise ValueError('batch_size는 양수')
    Path(args.models).mkdir(parents=True,exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('Sequence 학습 장치:',device)
    train_all,dev,_,cats,nums,_=read_v3(args.data)
    train_used=train_all.loc[train_mask(train_all)].copy()
    # First statistics only from allowed train rows; then transform ALL train inputs
    # so each sequence contains daylight and nighttime forecast context.
    _,xtrain,stats=standardize(train_used,train_all,nums)
    med=pd.Series(stats['medians']);mu=np.asarray(stats['mean'],np.float32);sd=np.asarray(stats['std'],np.float32)
    xdev=dev[nums].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy(dtype=np.float32,copy=True)
    xdev=(xdev-mu)/sd;np.clip(xdev,-8,8,out=xdev)
    _,ctrain,vocab=categoricals(train_used,train_all,cats)
    cdev=np.stack([dev[c].astype('string').fillna('__missing__').map(
              {v:i+1 for i,v in enumerate(vocab[c])}).fillna(0).to_numpy(dtype=np.int64,copy=True)
              for c in cats],axis=-1)
    tx,tc,ty,tm,_=prepare_sequence(train_all,xtrain,ctrain,train=True)
    vx,vc,vy,vm,devsorted=prepare_sequence(dev,xdev,cdev,train=False)
    print(f'Train 24h 시퀀스 {len(tx):,}개 (학습 마스크 {int(tm.sum()):,}시간) / Dev {len(vx):,}개')
    print('모델 차원 (수치,범주):',tx.shape[-1],tc.shape[-1])
    dims=[16 if c=='site_id' else 8 for c in cats]
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cat_emb=nn.ModuleList([nn.Embedding(len(vocab[c])+1,d,padding_idx=0)
                         for c,d in zip(cats,dims)])
            self.project=nn.Linear(len(nums)+sum(dims),64)
            self.position=nn.Embedding(24,64)
            if args.model=='transformer':
                block=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=256,
                                                dropout=.1,batch_first=True)
                self.backbone=nn.TransformerEncoder(block,num_layers=2)
                outdim=64
            else:
                # Full day-ahead forecast sequence is available: bidirectional context is legal.
                self.backbone=nn.LSTM(input_size=64,hidden_size=64,num_layers=2,
                                      bidirectional=True,dropout=.1,batch_first=True)
                outdim=128
            self.out=nn.Linear(outdim,1)
        def forward(self,num,cat):
            h=torch.cat([num]+[emb(cat[:,:,i]) for i,emb in enumerate(self.cat_emb)],dim=-1)
            h=self.project(h)
            h=h+self.position(torch.arange(24,device=num.device))[None,:,:]
            if args.model=='lstm': h,_=self.backbone(h)
            else: h=self.backbone(h)
            return self.out(h).squeeze(-1)
    model=Net().to(device)
    train_dl=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(tc),
                     torch.from_numpy(ty),torch.from_numpy(tm.astype(np.float32))),
                     batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_ds=TensorDataset(torch.from_numpy(vx),torch.from_numpy(vc))
    val_dl=DataLoader(val_ds,batch_size=512,shuffle=False,num_workers=0)
    optim=torch.optim.AdamW(model.parameters(),lr=.0003)
    criterion=nn.SmoothL1Loss(beta=.05,reduction='none')
    def infer():
        model.eval();parts=[]
        with torch.no_grad():
            for a,b in val_dl:
                pred=model(a.to(device),b.to(device)).float().cpu().numpy()
                parts.append(pred)
        return np.concatenate(parts,axis=0).reshape(-1)
    actual=devsorted.energy_kwh.to_numpy(dtype=np.float64)
    cap=devsorted.capacity_kw.to_numpy(dtype=np.float64)
    day=(devsorted.is_daylight.to_numpy()==1) if 'is_daylight' in devsorted else (devsorted.elev.to_numpy(dtype=float)>0)
    best=float('inf');wait=0;best_epoch=0;best_state=None;history=[];t0=time.time()
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;n=0
        for a,b,y,mask in train_dl:
            a,b,y,mask=a.to(device),b.to(device),y.to(device),mask.to(device)
            optim.zero_grad(set_to_none=True)
            raw=criterion(model(a,b),y)
            loss=(raw*mask).sum()/mask.sum().clamp(min=1)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optim.step()
            total+=float((raw*mask).sum().item());n+=int(mask.sum().item())
        cf=infer()
        pred=np.where(day,np.clip(cf,0,None)*cap,0.)
        mae=float(np.mean(np.abs(pred-actual)))
        history.append({'epoch':epoch,'train_loss':total/max(n,1),'dev_mae_kwh':mae})
        print(f'epoch {epoch:02d}/{args.epochs} loss={total/max(n,1):.5f} dev_MAE={mae:.4f}',flush=True)
        if mae < best-1e-5:
            best=mae;best_epoch=epoch;wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=args.patience: print('Early stopping'); break
    model.load_state_dict(best_state)
    cf=infer()
    pred=np.maximum(cf,0)*cap
    Path(args.outputs).mkdir(parents=True,exist_ok=True)
    pd.DataFrame(history).to_csv(Path(args.outputs)/f'{EXPS[args.model]}_history.csv',index=False)
    torch.save({'state_dict':best_state,'model_type':args.model,'num_features':nums,'cat_features':cats,
                'stats':stats,'vocab':vocab,'best_epoch':best_epoch,'data_version':'v0.4',
                'sequence_definition':'KST 01:00 through next 00:00, right-closed'},
               Path(args.models)/f'{EXPS[args.model]}.pt')
    result_files(EXPS[args.model],args.outputs,args.models,devsorted,pred,
               {'best_epoch':best_epoch,'train_seconds':round(time.time()-t0,1),
                'train_sequence_count':len(tx),'train_masked_row_count':int(tm.sum()),
                'numeric_features':nums,'categorical_features':cats,'device':device})

if __name__=='__main__':main()
