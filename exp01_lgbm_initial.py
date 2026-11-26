
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# ==========================================
# 1. 실험 및 환경 설정
# ==========================================

# EXP-01: LightGBM 초기 모델
EXP_ID = "exp01_lgbm_initial"
DATA_VERSION = "v0.1"

DATA_PATH = "data/train_v0.1.parquet"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print(f"실험 ID: {EXP_ID}")
print(f"데이터 버전: {DATA_VERSION}")
print("모델: LightGBM (Initial)")
print("학습 장치: CPU")
print("=" * 50)

# 학습·검증 기간 (KST)
# 팀 합의 전 임시 검증 구간

TRAIN_START = pd.Timestamp(
    "2024-06-01 01:00", tz="Asia/Seoul"
)
TRAIN_END = pd.Timestamp(
    "2025-02-27 01:00", tz="Asia/Seoul"
)

DEV_START = pd.Timestamp(
    "2025-03-01 01:00", tz="Asia/Seoul"
)
DEV_END = pd.Timestamp(
    "2025-05-01 01:00", tz="Asia/Seoul"
)

# ==========================================
# 2. 모델 입력 특징
# ==========================================

CAT_FEATURES = [
    "site_id",
    "wa_station",
    "wb_station",
]

NUM_FEATURES = [
    # 발전소 정보
    "capacity_kw", "lat", "lon",
    "wa_dist", "wb_dist",

    # 시간 특징
    "hour_sin", "hour_cos",
    "doy_sin", "doy_cos",

    # 태양 위치
    "elev", "azim", "cosz",
    "is_daylight",

    # A사 기상예보
    "wa_cloud_r10", "wa_cloud_r17",
    "wa_temp_r10", "wa_temp_r17",
    "wa_humidity_r10", "wa_humidity_r17",
    "wa_wind_speed_r10", "wa_wind_speed_r17",
    "wa_rain_r10", "wa_rain_r17",

    # B사 기상예보
    "wb_cloud_r10", "wb_cloud_r17",
    "wb_temp_r10", "wb_temp_r17",
    "wb_rel_hum_r10", "wb_rel_hum_r17",
    "wb_uv_idx_r10", "wb_uv_idx_r17",
    "wb_precip_prob_r10", "wb_precip_prob_r17",
    "wb_total_liq_r10", "wb_total_liq_r17",
    "wb_wind_speed_ms_r10", "wb_wind_speed_ms_r17",

    # 예보 결측 및 대체 표시
    "fc_filled_a", "fc_filled_b",
    "fc_missing_a", "fc_missing_b",
]

FEATURES = CAT_FEATURES + NUM_FEATURES

# 필요한 컬럼만 로딩하여 메모리 절약
LOAD_COLUMNS = list(dict.fromkeys(
    FEATURES + [
        "ts", "cf_target", "energy_kwh",
        "smp", "k"
    ]
))

print(f"\n사용할 입력 특징 수: {len(FEATURES)}")

# ==========================================
# 3. 데이터 로딩
# ==========================================

print("\n[1/7] 데이터 불러오는 중...")

df = pd.read_parquet(
    DATA_PATH,
    columns=LOAD_COLUMNS
)

print("전체 데이터:", df.shape)

# 이번 실험에 필요한 기간만 유지
df = df.loc[
    (df["ts"] >= TRAIN_START) &
    (df["ts"] < DEV_END)
].copy()

# 범주형 변수 변환
for col in CAT_FEATURES:
    df[col] = df[col].astype("category")

print("선택한 데이터:", df.shape)

# ==========================================
# 4. 시간순 학습·검증 분할
# ==========================================

print("\n[2/7] 학습·검증 데이터 분리...")

train_mask = (
    (df["ts"] >= TRAIN_START) &
    (df["ts"] < TRAIN_END)
)

dev_mask = (
    (df["ts"] >= DEV_START) &
    (df["ts"] < DEV_END)
)

train = df.loc[train_mask].copy()
dev = df.loc[dev_mask].copy()

del df

# 데이터 검증
assert len(train) > 0, "학습 데이터가 비어 있습니다."
assert len(dev) > 0, "검증 데이터가 비어 있습니다."
assert train["ts"].max() < dev["ts"].min()

print("\n===== 데이터 분할 결과 =====")
print("학습 데이터:", train.shape)
print("검증 데이터:", dev.shape)
print("학습 발전소:", train["site_id"].nunique())
print("검증 발전소:", dev["site_id"].nunique())

print(
    "학습 기간:",
    train["ts"].min(),
    "~",
    train["ts"].max()
)

print(
    "검증 기간:",
    dev["ts"].min(),
    "~",
    dev["ts"].max()
)

X_train = train[FEATURES]
y_train = train["cf_target"]

X_dev = dev[FEATURES]
y_dev = dev["cf_target"]

# ==========================================
# 5. LightGBM 초기 모델 학습
# ==========================================

print("\n[3/7] LightGBM 초기 모델 학습 시작...")

model = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    max_bin=127,
    n_jobs=4,
    random_state=42,
    verbosity=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_dev, y_dev)],
    eval_metric="l1",
    categorical_feature=CAT_FEATURES,
    callbacks=[
        lgb.early_stopping(40),
        lgb.log_evaluation(50),
    ]
)

print("\n학습 완료!")
print("최적 반복 횟수:", model.best_iteration_)

# ==========================================
# 6. 검증 발전량 예측
# ==========================================

print("\n[4/7] 검증 데이터 예측 중...")

pred_cf = model.predict(
    X_dev,
    num_iteration=model.best_iteration_
)

# 정규화 발전량 -> 실제 발전량(kWh)
pred_kwh = (
    np.maximum(pred_cf, 0)
    * dev["capacity_kw"].to_numpy()
)

# 야간 발전량 0 처리
pred_kwh = np.where(
    dev["is_daylight"].to_numpy() == 0,
    0.0,
    pred_kwh
)

actual_kwh = dev["energy_kwh"].to_numpy()
abs_error = np.abs(pred_kwh - actual_kwh)

# ==========================================
# 7. 검증 성능 계산
# ==========================================

print("\n[5/7] 검증 성능 계산 중...")

mae = np.mean(abs_error)

rmse = np.sqrt(
    np.mean((pred_kwh - actual_kwh) ** 2)
)

day_mask = (
    dev["is_daylight"].to_numpy() == 1
)

day_mae = np.mean(abs_error[day_mask])

print("\n===== EXP-01 초기 모델 검증 결과 =====")
print(f"MAE: {mae:.4f} kWh")
print(f"RMSE: {rmse:.4f} kWh")
print(f"주간 MAE: {day_mae:.4f} kWh")

# 발전소별 가중오차 참고값
# 실제 그룹별 정산 페널티와는 다름

weights = (
    dev["smp"].to_numpy()
    * dev["k"].to_numpy()
)

site_penalty_proxy = 0.10 * np.sum(
    weights * abs_error
)

print(
    "발전소별 가중오차 참고값:",
    f"{site_penalty_proxy:,.2f}"
)

# ==========================================
# 8. 초기 모델 저장
# ==========================================

print("\n[6/7] 초기 모델 저장 중...")

MODEL_PATH = MODEL_DIR / f"{EXP_ID}.txt"

model.booster_.save_model(
    str(MODEL_PATH)
)

print(f"모델 저장 완료: {MODEL_PATH}")

# ==========================================
# 9. 발전소별 예측 CSV 저장
# ==========================================

print("\n[7/7] 예측 결과 저장 중...")

predictions = dev[["site_id", "ts"]].copy()
predictions["pred_kwh"] = pred_kwh

predictions = predictions.sort_values(
    ["site_id", "ts"]
)

# 예측 파일 검증
assert not predictions.duplicated(
    ["site_id", "ts"]
).any(), "중복된 발전소·시각이 존재합니다."

assert predictions["pred_kwh"].notna().all()
assert np.isfinite(predictions["pred_kwh"]).all()
assert (predictions["pred_kwh"] >= 0).all()

# ------------------------------------------
# 9-1. 발전소별 예측값 저장
# ------------------------------------------

PRED_PATH = (
    OUTPUT_DIR / f"{EXP_ID}_dev_predictions.csv"
)

predictions.to_csv(
    PRED_PATH,
    index=False
)

# ------------------------------------------
# 9-2. 발전소별 예측 오차 저장
# ------------------------------------------

errors = predictions.merge(
    dev[["site_id", "ts", "energy_kwh"]],
    on=["site_id", "ts"],
    validate="one_to_one"
)

errors = errors.rename(
    columns={"energy_kwh": "actual_kwh"}
)

errors["error_kwh"] = (
    errors["pred_kwh"] - errors["actual_kwh"]
)

ERROR_PATH = (
    OUTPUT_DIR / f"{EXP_ID}_dev_errors.csv"
)

errors[
    [
        "site_id",
        "ts",
        "actual_kwh",
        "pred_kwh",
        "error_kwh",
    ]
].to_csv(
    ERROR_PATH,
    index=False
)

# ==========================================
# 10. 최종 실행 결과
# ==========================================

print("\n" + "=" * 50)
print("EXP-01 LightGBM 초기 모델 실행 완료")
print("=" * 50)

print(f"실험 ID: {EXP_ID}")
print(f"데이터 버전: {DATA_VERSION}")
print(f"최적 반복 횟수: {model.best_iteration_}")

print("\n[검증 성능]")
print(f"MAE: {mae:.4f} kWh")
print(f"RMSE: {rmse:.4f} kWh")
print(f"주간 MAE: {day_mae:.4f} kWh")

print("\n[저장 파일]")
print(f"모델: {MODEL_PATH}")
print(f"예측: {PRED_PATH}")
print(f"오차: {ERROR_PATH}")

print("\n모든 작업이 완료되었습니다.")
