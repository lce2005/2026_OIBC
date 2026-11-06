"""EXP-01 LightGBM: same 44 features, expanded training since 2023-06.
This is a new version, not a hyperparameter-tuning experiment.
Run in the VS Code 2026_OIBC project root:
    python exp01_lgbm_train2306.py
"""
from __future__ import annotations
import gc, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

EXP_ID = "exp01_lgbm_train2306"
DATA_VERSION = "v0.1"
DATA_PATH = Path("data/train_v0.1.parquet")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
TRAIN_START = pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul")
TRAIN_END = pd.Timestamp("2025-02-27 01:00", tz="Asia/Seoul")
DEV_START = pd.Timestamp("2025-03-01 01:00", tz="Asia/Seoul")
DEV_END = pd.Timestamp("2025-05-01 01:00", tz="Asia/Seoul")

CAT_FEATURES = ["site_id", "wa_station", "wb_station"]
NUM_FEATURES = [
    "capacity_kw", "lat", "lon", "wa_dist", "wb_dist",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "elev", "azim", "cosz", "is_daylight",
    "wa_cloud_r10", "wa_cloud_r17",
    "wa_temp_r10", "wa_temp_r17",
    "wa_humidity_r10", "wa_humidity_r17",
    "wa_wind_speed_r10", "wa_wind_speed_r17",
    "wa_rain_r10", "wa_rain_r17",
    "wb_cloud_r10", "wb_cloud_r17",
    "wb_temp_r10", "wb_temp_r17",
    "wb_rel_hum_r10", "wb_rel_hum_r17",
    "wb_uv_idx_r10", "wb_uv_idx_r17",
    "wb_precip_prob_r10", "wb_precip_prob_r17",
    "wb_total_liq_r10", "wb_total_liq_r17",
    "wb_wind_speed_ms_r10", "wb_wind_speed_ms_r17",
    "fc_filled_a", "fc_filled_b", "fc_missing_a", "fc_missing_b",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
LOAD_COLUMNS = list(dict.fromkeys(
    FEATURES + ["ts", "cf_target", "energy_kwh", "smp", "k"]
))



PARAMS = dict(
    objective="regression_l1",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_bin=127,
    n_jobs=4,
    random_state=42,
    verbosity=-1,
)


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"입력 데이터 없음: {DATA_PATH}")
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    started=time.perf_counter()
    frame=pd.read_parquet(DATA_PATH, columns=LOAD_COLUMNS)
    train=frame.loc[(frame.ts>=TRAIN_START)&(frame.ts<TRAIN_END)].copy()
    dev=frame.loc[(frame.ts>=DEV_START)&(frame.ts<DEV_END)].copy()
    del frame
    gc.collect()
    if train.empty or dev.empty:
        raise ValueError("학습/검증 구간이 비어 있습니다. 데이터의 시각 범위를 확인하세요.")
    for name,d in (("train",train),("dev",dev)):
        if d.duplicated(["site_id","ts"]).any():
            raise ValueError(f"{name}: 발전소·시간 중복")
    if train.ts.max()>=dev.ts.min():
        raise ValueError("Train/Dev 중복")
    for c in CAT_FEATURES:
        # Train and dev categories share a consistent vocabulary.
        categories=pd.Index(train[c].dropna().unique()).tolist()
        train[c]=pd.Categorical(train[c],categories=categories)
        dev[c]=pd.Categorical(dev[c],categories=categories)
    for c in NUM_FEATURES:
        train[c]=pd.to_numeric(train[c],errors="coerce").astype("float32")
        dev[c]=pd.to_numeric(dev[c],errors="coerce").astype("float32")
        train[c]=train[c].replace([np.inf,-np.inf],np.nan)
        dev[c]=dev[c].replace([np.inf,-np.inf],np.nan)
    ytrain=train.cf_target.to_numpy(dtype=np.float32,copy=True)
    ydev=dev.cf_target.to_numpy(dtype=np.float32,copy=True)
    if not np.isfinite(ytrain).all() or not np.isfinite(ydev).all():
        raise ValueError("cf_target에 NaN/Inf")
    print(f"[{EXP_ID}] Train {len(train):,}행 ({train.ts.min()} ~ {train.ts.max()})",flush=True)
    print(f"Dev {len(dev):,}행 ({dev.ts.min()} ~ {dev.ts.max()}) / 특성 {len(FEATURES)}개",flush=True)
    model=lgb.LGBMRegressor(**PARAMS)
    model.fit(
        train[FEATURES],ytrain,
        eval_set=[(dev[FEATURES],ydev)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50,verbose=True),lgb.log_evaluation(50)],
    )
    cf=model.predict(dev[FEATURES],num_iteration=model.best_iteration_)
    capacity=dev.capacity_kw.to_numpy(dtype=np.float64)
    pred=np.maximum(cf,0.0)*capacity
    pred=np.where(dev.is_daylight.to_numpy()==0,0.0,pred)
    actual=dev.energy_kwh.to_numpy(dtype=np.float64)
    if not np.isfinite(pred).all() or not np.isfinite(actual).all():
        raise ValueError("예측이나 실제값에 NaN/Inf")
    err=pred-actual
    maes=np.abs(err)
    day_mask=dev.is_daylight.to_numpy()==1
    weights=dev.smp.to_numpy(dtype=np.float64)*dev.k.to_numpy(dtype=np.float64)
    metrics={
        "experiment":EXP_ID,"data_version":DATA_VERSION,
        "train_start":str(TRAIN_START),"train_end_exclusive":str(TRAIN_END),
        "dev_start":str(DEV_START),"dev_end_exclusive":str(DEV_END),
        "train_rows":int(len(train)),"dev_rows":int(len(dev)),
        "features":FEATURES,"params":PARAMS,"best_iteration":int(model.best_iteration_),
        "mae_kwh":float(np.mean(maes)),
        "rmse_kwh":float(np.sqrt(np.mean(err**2))),
        "daylight_mae_kwh":float(np.mean(maes[day_mask])) if day_mask.any() else None,
        "site_weighted_penalty_proxy":float(0.1*np.sum(weights*maes)),
        "warning":"site-weighted proxy is NOT official grouped VPP penalty",
        "elapsed_seconds":round(time.perf_counter()-started,2),
    }
    model_path=MODEL_DIR/f"{EXP_ID}.txt"
    model.booster_.save_model(str(model_path))
    errors=dev[["site_id","ts"]].copy()
    errors["actual_kwh"]=actual
    errors["pred_kwh"]=pred
    errors["error_kwh"]=err
    errors=errors.sort_values(["site_id","ts"])
    errors[["site_id","ts","pred_kwh"]].to_csv(OUTPUT_DIR/f"{EXP_ID}_dev_predictions.csv",index=False)
    errors.to_csv(OUTPUT_DIR/f"{EXP_ID}_dev_errors.csv",index=False)
    (OUTPUT_DIR/f"{EXP_ID}_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[완료] MAE={metrics['mae_kwh']:.4f} / RMSE={metrics['rmse_kwh']:.4f} kWh")
    print(f"[저장] {model_path} 및 outputs/{EXP_ID}_dev_*.csv")

if __name__=="__main__":
    main()
