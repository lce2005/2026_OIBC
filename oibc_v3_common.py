"""OIBC v0.4 shared data contract. Train only on split=train & use_train=1.
Dev always evaluates all rows. 00:00 belongs to preceding delivery day.
No observed weather, actual energy, split, or price is ever used as an input.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_BEGIN = pd.Timestamp('2023-06-01 01:00', tz='Asia/Seoul')
DEV_BEGIN = pd.Timestamp('2025-03-01 01:00', tz='Asia/Seoul')
HOLD_BEGIN = pd.Timestamp('2025-05-01 01:00', tz='Asia/Seoul')
HOLD_END = pd.Timestamp('2025-06-01 01:00', tz='Asia/Seoul')
CAT_CANDIDATES = ['site_id', 'wa_station', 'wb_station']
# Explicitly allowlisted pre-forecast and static covariates. Never automatically
# add arbitrary numeric columns; this prevents data leakage from new v0.4 columns.
NUM_CANDIDATES = [
    'capacity_kw','lat','lon','wa_dist','wb_dist',
    'hour_sin','hour_cos','doy_sin','doy_cos',
    'elev','azim','cosz','is_daylight',
    'wa_cloud_r10','wa_cloud_r17','wa_temp_r10','wa_temp_r17',
    'wa_humidity_r10','wa_humidity_r17','wa_wind_speed_r10','wa_wind_speed_r17',
    'wa_rain_r10','wa_rain_r17',
    'wb_cloud_r10','wb_cloud_r17','wb_temp_r10','wb_temp_r17',
    'wb_rel_hum_r10','wb_rel_hum_r17','wb_uv_idx_r10','wb_uv_idx_r17',
    'wb_precip_prob_r10','wb_precip_prob_r17',
    'wb_total_liq_r10','wb_total_liq_r17',
    'wb_wind_speed_ms_r10','wb_wind_speed_ms_r17',
    'fc_filled_a','fc_filled_b','fc_missing_a','fc_missing_b',
    # Optional v0.4-derived forecasts; included only if truly present:
    'wa_ground_press_r10','wa_ground_press_r17',
    'wb_vis_r10','wb_vis_r17','wb_dew_point_r10','wb_dew_point_r17',
    'wa_snow_r10','wa_snow_r17','wb_snow_r10','wb_snow_r17',
]
REQUIRED = ['site_id','ts','split','use_train','capacity_kw','energy_kwh','cf_target']

def find_data(path=None):
    if path:
        p = Path(path).expanduser()
        if not p.is_file(): raise FileNotFoundError(f'v0.4 parquet를 찾을 수 없습니다: {p}')
        return p
    for name in ['data/train_v0.4.parquet', 'data/data_v0.4/train_v0.4.parquet',
                 'data/v0.4/train_v0.4.parquet', 'data/Data/data_v0.4/train_v0.4.parquet']:
        p = Path(name)
        if p.is_file(): return p
    raise FileNotFoundError('v0.4 parquet 위치를 확인하세요: --data data/data_v0.4/train_v0.4.parquet')

def get_schema(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as ex:
        raise RuntimeError('pyarrow 설치 필요: python -m pip install pyarrow') from ex
    return pq.read_schema(str(path)).names

def to_kst(series):
    raw = pd.to_datetime(series, errors='raise')
    if getattr(raw.dt, 'tz', None) is None:
        # Competition time notation is KST for naive source timestamps.
        return raw.dt.tz_localize('Asia/Seoul')
    return raw.dt.tz_convert('Asia/Seoul')

def feature_names(schema):
    missing = sorted(set(REQUIRED) - set(schema))
    if missing: raise ValueError('v0.4 필수 열 누락: '+str(missing)+'. 실제 컬럼명을 확인하세요.')
    nums = [c for c in NUM_CANDIDATES if c in schema]
    cats = [c for c in CAT_CANDIDATES if c in schema]
    if 'site_id' not in cats or 'capacity_kw' not in nums:
        raise ValueError('site_id/capacity_kw가 필요합니다')
    if len(nums) < 20: raise ValueError(f'허용된 수치 입력 특징이 너무 적습니다: {len(nums)}. v0.4 스키마 확인 필요.')
    return cats, nums

def read_v3(path=None, *, holdout=False):
    path = find_data(path)
    schema = get_schema(path)
    cats, nums = feature_names(schema)
    columns = list(dict.fromkeys(REQUIRED + cats + nums))
    df = pd.read_parquet(path, columns=columns)
    df['ts'] = to_kst(df['ts'])
    # Enforce the TEAM's agreed clock boundaries even if split column is wrong.
    df = df.loc[(df.ts >= TRAIN_BEGIN) & (df.ts < (HOLD_END if holdout else HOLD_BEGIN))].copy()
    label = df['split'].astype('string').str.lower().str.strip()
    m_train = (df.ts < DEV_BEGIN)
    m_dev = (df.ts >= DEV_BEGIN) & (df.ts < HOLD_BEGIN)
    if (label[m_train] != 'train').any():
        raise ValueError('Train 시간 범위와 split=train 값이 일치하지 않습니다.')
    if not label[m_dev].isin(['dev','val','valid','validation']).all():
        raise ValueError('Dev 시간 범위와 split=dev/val 값이 일치하지 않습니다. 실제 split 값 확인 필요: '+str(label[m_dev].value_counts().to_dict()))
    train_all = df.loc[m_train].copy()
    dev = df.loc[m_dev].copy()
    hold = df.loc[(df.ts >= HOLD_BEGIN) & (df.ts < HOLD_END)].copy() if holdout else None
    if train_all.empty or dev.empty: raise ValueError('Train 또는 Dev 데이터가 비어 있습니다.')
    for name, part in [('train',train_all),('dev',dev)]:
        if part.duplicated(['site_id','ts']).any(): raise ValueError(f'{name}: site_id-ts 중복')
        if part['energy_kwh'].isna().any(): raise ValueError(f'{name}: energy_kwh 결측')
    if len(dev) != 253*1464 or dev.site_id.nunique() != 253:
        raise ValueError(f'Dev는 전체 253*1464=370392행이어야 합니다. 실제 {len(dev)}행 / {dev.site_id.nunique()}개소')
    if dev.groupby('ts',observed=True).size().nunique() != 1:
        raise ValueError('Dev 일부 시간대의 발전소 행이 누락되어 있습니다')
    if (train_all['use_train'].astype('string') == '1').sum() == 0:
        raise ValueError('use_train=1 행이 없습니다. v0.4 열 값을 확인하세요')
    print(f'[v0.4] 파일={path} | train 원본 {len(train_all):,} | train 사용 {int((train_all.use_train.astype("string") == "1").sum()):,} | dev 전체 {len(dev):,}')
    print(f'[v0.4] 특징 {len(cats)+len(nums)}개 (범주 {len(cats)}, 수치 {len(nums)})')
    print('[v0.4] 선택 특징:', cats + nums)
    return train_all, dev, hold, cats, nums, path

def train_mask(df):
    return df['use_train'].astype('string').eq('1').to_numpy()

def standardize(train, dev, num):
    # Learn medians, centers, scales exclusively from permitted TRAIN rows.
    tr = train[num].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float32')
    med = tr.median().fillna(0.0).astype('float32')
    a = tr.fillna(med).to_numpy(dtype=np.float32, copy=True)
    mean = a.mean(axis=0,dtype=np.float64).astype('float32')
    sd = a.std(axis=0,dtype=np.float64).astype('float32')
    sd[sd<1e-6]=1.0
    def transform(frame):
        a = frame[num].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(med).to_numpy(dtype=np.float32, copy=True)
        a -= mean; a /= sd; np.clip(a,-8,8,out=a)
        return a
    return transform(train), transform(dev), {'medians':med.to_dict(),'mean':mean.tolist(),'std':sd.tolist()}

def categoricals(train, dev, cats):
    left=[]; right=[]; vocabulary={}
    for col in cats:
        labels = train[col].astype('string').fillna('__missing__')
        vocab = sorted(labels.unique().tolist())
        mapping={v:i+1 for i,v in enumerate(vocab)}
        left.append(labels.map(mapping).to_numpy(dtype=np.int64,copy=True))
        right.append(dev[col].astype('string').fillna('__missing__').map(mapping).fillna(0).to_numpy(dtype=np.int64,copy=True))
        vocabulary[col]=vocab
    return np.stack(left,axis=1), np.stack(right,axis=1), vocabulary

def result_files(exp, out, model, dev, predictions, extras):
    out=Path(out); model=Path(model); out.mkdir(exist_ok=True,parents=True); model.mkdir(exist_ok=True,parents=True)
    preds=np.asarray(predictions,dtype='float64')
    if len(preds)!=len(dev) or not np.isfinite(preds).all(): raise ValueError('Dev 예측값 길이/유한성 오류')
    pred=np.clip(preds,0,None)
    if 'is_daylight' in dev:
        pred=np.where(dev['is_daylight'].to_numpy().astype(bool),pred,0)
    elif 'elev' in dev:
        pred=np.where(dev['elev'].to_numpy(dtype=float)>0,pred,0)
    actual=dev.energy_kwh.to_numpy(dtype=np.float64)
    err=pred-actual
    frame=dev[['site_id','ts']].copy()
    frame['actual_kwh']=actual; frame['pred_kwh']=pred; frame['error_kwh']=err
    frame=frame.sort_values(['site_id','ts'])
    frame[['site_id','ts','pred_kwh']].to_csv(out/f'{exp}_dev_predictions.csv',index=False)
    frame.to_csv(out/f'{exp}_dev_errors.csv',index=False)
    daylight=(dev['elev'].to_numpy(dtype=float)>0) if 'elev' in dev else (dev['is_daylight'].to_numpy()==1)
    metrics={**extras, 'experiment':exp, 'data_version':'v0.4', 'train_start_kst':str(TRAIN_BEGIN),
             'train_end_exclusive_kst':str(DEV_BEGIN), 'dev_start_kst':str(DEV_BEGIN),
             'dev_end_exclusive_kst':str(HOLD_BEGIN), 'dev_rows':len(dev),
             'mae_kwh':float(np.mean(np.abs(err))), 'rmse_kwh':float(np.sqrt(np.mean(err**2))),
             'bias_kwh':float(np.mean(err)), 'daylight_mae_kwh':float(np.mean(np.abs(err[daylight])))}
    (out/f'{exp}_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(f'[완료 {exp}] MAE={metrics["mae_kwh"]:.4f}, RMSE={metrics["rmse_kwh"]:.4f}, 주간 MAE={metrics["daylight_mae_kwh"]:.4f} kWh')
    print(f'저장: {out}/{exp}_dev_errors.csv')
    return metrics
