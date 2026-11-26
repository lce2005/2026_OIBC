"""OIBC 2026 | 참가자용 baseline.py를 EXP-01/02/03과 같은 Dev 기간에서 검증.

**주의** 원본 baseline.py를 그대로 실행한 결과가 아니라, 해당 코드의 FEATS,
features(), 두 LightGBM(라운드 10/17), 예측 평균 로직을 보존하면서
우리 팀의 train_v0.1 시간순 분할에 맞춘 *검증용 변형*이다.

요구 파일 (프로젝트 root):
    baseline.py  [사용자가 받은 참가자용 원본; 변경하지 않음]
    data/train_v0.1.parquet  [이주영 전처리 데이터]
    data/sites.csv          [원본 사이트 메타]
    data/raw/train/forecast_wa.parquet   [원본 A 기상 예보]
    data/raw/train/forecast_wb.parquet   [원본 B 기상 예보]
    outputs/exp01_lgbm_initial_dev_errors.csv [동일 dev 키/정답 확보]

대회 원본 파일들을 다른 위치에 보관한다면 --raw-root 경로를 지정.
  python exp00_official_baseline_dev.py --raw-root data/raw
  python exp00_official_baseline_dev.py --raw-root /path/to/official/dataset

원본 제출용 코드는 'python baseline.py' 이며, 본 파일은 제출 파일을 만들지 않는다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import baseline as official

TRAIN_START = pd.Timestamp("2024-06-01 01:00", tz="Asia/Seoul")
TRAIN_END = pd.Timestamp("2025-02-27 01:00", tz="Asia/Seoul")
DEV_START = pd.Timestamp("2025-03-01 01:00", tz="Asia/Seoul")
DEV_END = pd.Timestamp("2025-05-01 01:00", tz="Asia/Seoul")


def as_kst(s: pd.Series) -> pd.Series:
    # 원본 타임스탬프는 tz-aware로 제공됨. Naive를 임의로 UTC 처리하지 않는다.
    ts = pd.to_datetime(s, errors="raise")
    if ts.dt.tz is None:
        raise ValueError("ts에 시간대 정보가 없습니다. 원본의 KST/UTC 설정 확인이 필요합니다.")
    return ts.dt.tz_convert("Asia/Seoul")


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"필수 입력 파일이 없습니다: {path}")
    return path


def make_model() -> lgb.LGBMRegressor:
    # 공식 baseline.py의 설정을 그대로 보존.
    return lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.06,
        num_leaves=127,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        n_jobs=-1,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--raw-root", type=Path, default=Path("data/raw"),
                    help="원본 데이터 루트: sites.csv 또는 train/forecast_wa.parquet 등")
    args = ap.parse_args()
    root = args.root.resolve()
    raw_root = args.raw_root
    if not raw_root.is_absolute():
        raw_root = (root / raw_root).resolve()

    t0 = time.perf_counter()
    out_dir = root / "outputs"
    model_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # 원본 사이트 메타는 data/sites.csv가 있으면 우선 사용하고, 없으면 raw-root/sites.csv
    sites_path = root / "data" / "sites.csv"
    if not sites_path.is_file():
        sites_path = raw_root / "sites.csv"
    sites = pd.read_csv(require(sites_path), usecols=[
        "site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"
    ])
    sites["site_id"] = sites["site_id"].astype("string")
    if len(sites) != 253 or sites["site_id"].duplicated().any():
        raise ValueError("sites.csv에 253개 발전소의 고유 site_id가 필요합니다.")

    prepared_path = require(root / "data" / "train_v0.1.parquet")
    train = pd.read_parquet(prepared_path, columns=["site_id", "ts", "energy_kwh"])
    train["site_id"] = train["site_id"].astype("string")
    train["ts"] = as_kst(train["ts"])
    train = train.loc[(train.ts >= TRAIN_START) & (train.ts < TRAIN_END)]
    train = train.rename(columns={"energy_kwh": "energy"}).copy()
    if train.empty or train[["site_id", "ts", "energy"]].isna().any().any():
        raise ValueError("학습 데이터가 없거나 필요한 열에 결측이 존재합니다.")
    if train.duplicated(["site_id", "ts"]).any():
        raise ValueError("학습 데이터의 site_id,ts 키 중복")
    print(f"[1/5] Train {len(train):,}행, sites {len(sites)}개, features {len(official.FEATS)}개", flush=True)

    # 기존 EXP-01의 검증 키/실적을 그대로 사용해 네 모델 정답과 기간을 통일한다.
    ref_path = require(out_dir / "exp01_lgbm_initial_dev_errors.csv")
    ref = pd.read_csv(ref_path, usecols=["site_id", "ts", "actual_kwh"])
    ref["site_id"] = ref["site_id"].astype("string")
    ref["ts"] = as_kst(ref["ts"])
    if ref.duplicated(["site_id", "ts"]).any() or ref[["site_id", "ts", "actual_kwh"]].isna().any().any():
        raise ValueError("EXP-01 dev 오류 파일이 올바르지 않습니다")
    ref = ref.sort_values(["site_id", "ts"]).reset_index(drop=True)
    if (ref.ts < DEV_START).any() or (ref.ts >= DEV_END).any():
        raise ValueError("참조 파일 dev 기간이 일치하지 않습니다")
    times = pd.DatetimeIndex(sorted(ref.ts.unique()))
    if len(times) != 1464 or len(ref) != 370392 or set(ref.site_id) != set(sites.site_id):
        raise ValueError("EXP-01의 Dev 키가 253개 발전소 x 1464시간과 일치하지 않습니다")
    print(f"[2/5] Dev {len(ref):,}행 / 1,464시간 (기존 EXP-01 정답 사용)", flush=True)

    def load_forecast(system: str, cols: list[str]) -> pd.DataFrame:
        path = require(raw_root / "train" / f"forecast_{system}.parquet")
        station = "wa_station" if system == "wa" else "wb_station"
        d = pd.read_parquet(path, columns=[station, "round", "ts"] + cols)
        d["ts"] = as_kst(d["ts"])
        d = d.loc[(d.ts >= TRAIN_START) & (d.ts < DEV_END)]
        if d.duplicated([station, "round", "ts"]).any():
            raise ValueError(f"원본 예보 {path} station/round/ts 중복")
        return d

    fa = load_forecast("wa", official.WA_COLS)
    fb = load_forecast("wb", official.WB_COLS)
    print(f"[3/5] 원본 예보 A {len(fa):,}행 / B {len(fb):,}행", flush=True)

    forecasts = []
    for round_id in official.ROUNDS:
        print(f"[4/5] 라운드 {round_id}: 공식 기준선 LightGBM 학습", flush=True)
        tr = official.features(train, sites, fa, fb, None, round_id)
        if len(tr) != len(train):
            raise ValueError(f"라운드 {round_id}: 원본 예보 조인 시 행 수가 변했습니다")
        tr = tr.loc[tr.elev > 0]
        X_train = tr[official.FEATS].astype("float32")
        y_train = (tr.energy / tr.cap).clip(0, 1.2).astype("float32")
        if X_train.isna().all(axis=1).any() or not np.isfinite(y_train.to_numpy()).all():
            raise ValueError("학습 데이터에 전부 결측인 특성 행 또는 비정상 정답이 존재합니다")
        model = make_model()
        model.fit(X_train, y_train)
        model_path = model_dir / f"exp00_baseline_round{round_id}.txt"
        model.booster_.save_model(str(model_path))
        print(f"       학습 {len(tr):,}행, 저장 {model_path.name}", flush=True)
        del tr, X_train, y_train

        d = official.features(None, sites, fa, fb, times, round_id)
        if len(d) != len(ref):
            raise ValueError("Dev 전체 격자 생성 결과가 기존 모델과 다릅니다")
        X_dev = d[official.FEATS].astype("float32")
        p = model.predict(X_dev)
        kwh = np.clip(np.where(d.elev.to_numpy() > 0, p, 0.0), 0, None) * d.cap.to_numpy()
        forecasts.append(pd.DataFrame({
            "site_id": d.site_id.astype("string"), "ts": d.ts, "pred_kwh": kwh,
        }))
        del d, X_dev, model

    # 공식 baseline.py: 두 라운드 예측 평균
    preds = pd.concat(forecasts, ignore_index=True).groupby(
        ["site_id", "ts"], observed=True, as_index=False
    )["pred_kwh"].mean()
    merged = ref.merge(preds, on=["site_id", "ts"], how="left", validate="one_to_one")
    if len(merged) != len(ref) or merged.pred_kwh.isna().any():
        raise ValueError("Dev 전체 발전소/시간 예측이 누락되었습니다")
    merged = merged.sort_values(["site_id", "ts"]).reset_index(drop=True)
    if not np.isfinite(merged.pred_kwh).all():
        raise ValueError("비정상 예측값 NaN/Inf")
    merged["error_kwh"] = merged["pred_kwh"] - merged["actual_kwh"]
    preds_path = out_dir / "exp00_baseline_dev_predictions.csv"
    errors_path = out_dir / "exp00_baseline_dev_errors.csv"
    merged[["site_id", "ts", "pred_kwh"]].to_csv(preds_path, index=False)
    merged[["site_id", "ts", "actual_kwh", "pred_kwh", "error_kwh"]].to_csv(errors_path, index=False)

    # 기존 비교 코드의 채점 함수를 재사용하고, 같은 actual, same keys 체크.
    try:
        from compare_exp01_exp02_exp03 import read_error_csv, load_groups, load_price, score_model
    except ImportError as exc:
        raise ImportError("compare_exp01_exp02_exp03.py를 프로젝트 최상위에 두세요") from exc

    files = {
        "exp00_official_baseline_dev_adapted": errors_path,
        "exp01_lgbm_initial": out_dir / "exp01_lgbm_initial_dev_errors.csv",
        "exp02_mlp_initial": out_dir / "exp02_mlp_initial_dev_errors.csv",
        "exp03_xgb_gpu": out_dir / "exp03_xgb_gpu_dev_errors.csv",
    }
    assignment, caps, group_source = load_groups(sites_path, None)
    ref_utc = read_error_csv(ref_path)
    prices = load_price(prepared_path, ref_utc["ts"].drop_duplicates())
    results = []
    hourly = []
    for name, path in files.items():
        if not path.exists():
            print(f"     [참고] {path.name}가 없어 비교에서 제외")
            continue
        current = read_error_csv(path)
        if len(current) != len(ref_utc) or not current[["site_id", "ts"]].equals(ref_utc[["site_id", "ts"]]):
            raise ValueError(f"{name}: Dev 키가 EXP-01과 일치하지 않습니다")
        if not np.allclose(current.actual_kwh, ref_utc.actual_kwh, atol=1e-3, rtol=0):
            raise ValueError(f"{name}: 실제 발전량이 EXP-01과 다릅니다")
        score, group_hourly = score_model(current, assignment, prices, name)
        results.append(score)
        hourly.append(group_hourly)

    comparison = pd.DataFrame(results).sort_values("group_penalty_krw")
    comparison_path = out_dir / "exp00_vs_exp01_exp02_exp03_dev_scores.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    pd.concat(hourly, ignore_index=True).to_csv(
        out_dir / "exp00_vs_exp01_exp02_exp03_group_hourly_dev.csv", index=False,
        encoding="utf-8-sig"
    )
    baseline_score = next(s for s in results if s["experiment"] == "exp00_official_baseline_dev_adapted")
    info = {
        "name": "exp00_official_baseline_dev_adapted",
        "source": "baseline.py (공식 참가자용 기준선)에서 특징/모델/두 라운드 평균 로직 재사용",
        "important": "원본 baseline.py 자체의 점수가 아님: train_v0.1 및 공동 Dev 기간으로 변형하여 평가",
        "train_window_kst": f"[{TRAIN_START}, {TRAIN_END})",
        "dev_window_kst": f"[{DEV_START}, {DEV_END})",
        "num_features": len(official.FEATS),
        "num_models": len(official.ROUNDS),
        "group_source": group_source,
        "group_capacity_kw": {str(k): float(v) for k, v in caps.items()},
        "training_seconds": round(time.perf_counter() - t0, 3),
        "metrics": baseline_score,
    }
    (out_dir / "exp00_baseline_dev_metrics.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[5/5] 공식 베이스라인 *Dev 변형* vs EXP-01/02/03 비교")
    print(comparison[["experiment", "mae_kwh", "rmse_kwh", "group_penalty_krw", "settlement_krw"]].to_string(index=False))
    print("\n저장:", preds_path, errors_path, comparison_path, sep="\n  ")
    print("참고: original baseline.py는 데이터 누출 없이 테스트/제출 예측을 하는 프로그램이며, 여기서는 Dev 용도로 재구성했습니다.")


if __name__ == "__main__":
    main()
