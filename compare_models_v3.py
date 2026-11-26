"""Fair v0.4 Dev group settlement comparison across up to five saved models.
Unlike per-site proxy, aggregate errors within each group before absolute value.
No price/actuals enter forecasting features; score file is for evaluation only.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from oibc_v3_common import find_data,get_schema,to_kst,DEV_BEGIN,HOLD_BEGIN

EXPERIMENTS=['exp00_official_baseline_v3_dev_adapted','exp01_lgbm_v3','exp02_mlp_v3','exp03_xgb_gpu_v3',
             'exp05_transformer_v3','exp06_lstm_v3']

def read_errors(path):
    d=pd.read_csv(path,usecols=['site_id','ts','actual_kwh','pred_kwh'])
    d['site_id']=d.site_id.astype('string')
    d['ts']=pd.to_datetime(d.ts,utc=True).dt.tz_convert('Asia/Seoul')
    if d.duplicated(['site_id','ts']).any():raise ValueError(f'{path}: site/ts 중복')
    if d.isna().any().any():raise ValueError(f'{path}: 결측')
    if not np.isfinite(d[['actual_kwh','pred_kwh']].to_numpy(dtype=float)).all():
        raise ValueError(f'{path}: NaN/Inf')
    if (d.pred_kwh<0).any():raise ValueError(f'{path}: 음수 예측')
    if not d.ts.ge(DEV_BEGIN).all() or not d.ts.lt(HOLD_BEGIN).all():
        raise ValueError(f'{path}: Dev 밖의 데이터')
    return d.sort_values(['site_id','ts']).reset_index(drop=True)

def read_groups(sites_path,composition_path=None):
    s=pd.read_csv(sites_path)
    s.site_id=s.site_id.astype('string')
    if len(s)!=253 or s.site_id.duplicated().any():raise ValueError('sites.csv에 253개 발전소가 필요합니다.')
    if composition_path:
        g=pd.read_csv(composition_path,usecols=['site_id','group'])
        g.site_id=g.site_id.astype('string')
    else:
        g=s[['site_id','current_group']].rename(columns={'current_group':'group'}).copy()
        g['group']=g.group.map({'A':'G1','B':'G2'})
    if g.site_id.duplicated().any() or set(g.site_id)!=set(s.site_id):raise ValueError('배정 누락/중복')
    if set(g.group)!={'G1','G2'}: raise ValueError('그룹은 G1/G2')
    caps=g.merge(s[['site_id','capacity_kw']],on='site_id').groupby('group').capacity_kw.sum()
    if (caps<20000).any():raise ValueError(f'20,000kW 용량 조건 미달: {caps}')
    return g,caps.to_dict()

def read_price(data_path,smp_path=None):
    path=find_data(data_path)
    cols=get_schema(path)
    if {'ts','smp'}.issubset(cols):
        d=pd.read_parquet(path,columns=['ts','smp'])
    else:
        if smp_path is None:
            raise ValueError('v0.4에 smp가 없습니다. --smp data/raw/train/smp.csv 로 과거 SMP 파일을 지정하세요.')
        d=pd.read_csv(smp_path,usecols=['ts','smp'])
    d['ts']=to_kst(d.ts)
    d=d.loc[d.ts.ge(DEV_BEGIN)&d.ts.lt(HOLD_BEGIN),['ts','smp']].drop_duplicates()
    if d.ts.duplicated().any():raise ValueError('한 시각에 서로 다른 SMP가 있습니다')
    if d.smp.isna().any() or not np.isfinite(d.smp.to_numpy(dtype=float)).all():raise ValueError('SMP 결측/무한')
    d['k']=np.where(d.ts.dt.hour.between(11,15),3,1)
    if len(d)!=1464:raise ValueError(f'Dev 기간 SMP 1464시간 필요: 현재 {len(d)}')
    return d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default=None)
    ap.add_argument('--sites',default='data/raw/sites.csv')
    ap.add_argument('--composition',default=None)
    ap.add_argument('--smp',default=None)
    ap.add_argument('--outputs',default='outputs')
    ap.add_argument('--require-all',action='store_true')
    args=ap.parse_args()
    out=Path(args.outputs);out.mkdir(parents=True,exist_ok=True)
    gp,caps=read_groups(args.sites,args.composition)
    price=read_price(args.data,args.smp)
    reference=None;stats=[];hourly=[];used=[]
    for exp in EXPERIMENTS:
        path=out/f'{exp}_dev_errors.csv'
        if not path.exists():
            if args.require_all: raise FileNotFoundError(str(path))
            print('[건너뜀] 결과 파일 없음:',path);continue
        d=read_errors(path)
        if len(d)!=253*1464:raise ValueError(f'{exp}: 370,392행이 아닙니다.')
        if reference is None: reference=d[['site_id','ts','actual_kwh']].copy()
        else:
            if not d[['site_id','ts']].equals(reference[['site_id','ts']]):
                raise ValueError(f'{exp}: 다른 모델과 발전소/시간축 불일치')
            if not np.allclose(d.actual_kwh,reference.actual_kwh,rtol=0,atol=.001):
                raise ValueError(f'{exp}: 실제 발전량 불일치')
        if set(d.site_id)!=set(gp.site_id):raise ValueError(f'{exp}: 사이트 누락')
        m=d.merge(gp,on='site_id',validate='many_to_one')
        m['error_kwh']=m.pred_kwh-m.actual_kwh
        grp=m.groupby(['group','ts'],as_index=False).agg(pred_kwh=('pred_kwh','sum'),
                actual_kwh=('actual_kwh','sum'),error_kwh=('error_kwh','sum'),
                site_count=('site_id','nunique'))
        for g in ['G1','G2']:
            expected=(gp.group==g).sum()
            if not (grp.loc[grp.group==g,'site_count']==expected).all():raise ValueError(f'{exp}: 그룹 {g} 일부 시간 누락')
        grp=grp.merge(price,on='ts',validate='many_to_one')
        if grp.smp.isna().any() or len(grp)!=2*1464:raise ValueError(f'{exp}: 시각/가격 누락')
        grp['penalty_krw']=.10*grp.smp*grp.k*grp.error_kwh.abs()
        grp['revenue_krw']=grp.smp*grp.actual_kwh
        grp['experiment']=exp
        grp['peak']=grp.ts.dt.hour.between(11,15)
        err=m.error_kwh.to_numpy(dtype=float)
        penalty=float(grp.penalty_krw.sum())
        revenue=float(grp.revenue_krw.sum())
        stats.append({'experiment':exp,'rows':len(d),'mae_kwh':float(np.mean(abs(err))),
           'rmse_kwh':float(np.sqrt(np.mean(err**2))),
           'bias_kwh':float(np.mean(err)),'group_penalty_krw':penalty,
           'peak_penalty_krw':float(grp.loc[grp.peak,'penalty_krw'].sum()),
           'settlement_krw':revenue-penalty})
        hourly.append(grp)
        used.append(exp)
    if not used:raise ValueError('v0.4 모델 예측 결과 파일이 하나도 없습니다')
    score=pd.DataFrame(stats).sort_values('group_penalty_krw').reset_index(drop=True)
    base='exp01_lgbm_v3'
    if base in used:
        baseval=float(score.loc[score.experiment==base,'group_penalty_krw'].iloc[0])
        score['penalty_improvement_vs_lgbm_pct']=(baseval-score.group_penalty_krw)/baseval*100
    score_path=out/'compare_models_v3_group_scores_dev.csv';hour_path=out/'compare_models_v3_group_hourly_dev.csv'
    score.to_csv(score_path,index=False)
    pd.concat(hourly,ignore_index=True).to_csv(hour_path,index=False)
    (out/'compare_models_v3_group_scores_dev_info.json').write_text(json.dumps({
      'data_version':'v0.4','period':'2025-03-01 01:00 <= ts < 2025-05-01 01:00 KST',
      'group_capacity_kw':caps,'composition':str(args.composition or 'sites.csv current_group'),
      'models':used,'note':'Historical dev only; 2025-05 holdout reserved.'
    },ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n=== v0.4 그룹 페널티 비교 (Dev) ===')
    print(score.to_string(index=False))
    print('저장:',score_path,hour_path)

if __name__=='__main__':main()
