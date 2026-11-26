"""OIBC 2026 | EXP-01/02/03 fair development-period comparison.

Inputs (project root):
  data/train_v0.1.parquet           : ts, smp, k
  data/sites.csv                    : site_id, current_group, capacity_kw
  outputs/exp01_lgbm_initial_dev_errors.csv
  outputs/exp02_mlp_initial_dev_errors.csv
  outputs/exp03_xgb_gpu_dev_errors.csv

Run:
  python compare_exp01_exp02_exp03.py
  python compare_exp01_exp02_exp03.py --composition data/composition.csv

Without --composition, the reference A/B assignment in sites.csv is mapped to G1/G2.
For a fair comparison, all models must have exactly the same (site_id, ts) keys,
actual_kwh, SMP, and group assignment.

This computes the README settlement formula on the historical dev period;
it is NOT a leaderboard/test score, and the group assignment is NOT optimized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = {
    "exp01_lgbm_initial": "exp01_lgbm_initial_dev_errors.csv",
    "exp02_mlp_initial": "exp02_mlp_initial_dev_errors.csv",
    "exp03_xgb_gpu": "exp03_xgb_gpu_dev_errors.csv",
}

DEV_START = pd.Timestamp("2025-03-01 01:00:00", tz="Asia/Seoul").tz_convert("UTC")
DEV_END = pd.Timestamp("2025-05-01 01:00:00", tz="Asia/Seoul").tz_convert("UTC")


def read_error_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"필수 예측 오차 파일이 없습니다: {path}")
    df = pd.read_csv(path, usecols=["site_id", "ts", "actual_kwh", "pred_kwh"])
    df["site_id"] = df["site_id"].astype("string")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="raise")

    if df[["site_id", "ts", "actual_kwh", "pred_kwh"]].isna().any().any():
        raise ValueError(f"필수 컬럼에 결측이 있습니다: {path}")
    if df.duplicated(["site_id", "ts"]).any():
        raise ValueError(f"(site_id, ts) 중복 행이 있습니다: {path}")
    if not np.isfinite(df[["actual_kwh", "pred_kwh"]].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"실제값 또는 예측값이 NaN/Inf입니다: {path}")
    if (df["pred_kwh"] < 0).any():
        raise ValueError(f"음수 예측이 존재합니다: {path}")
    if ((df["ts"] < DEV_START) | (df["ts"] >= DEV_END)).any():
        raise ValueError(f"합의한 3~4월 Dev 기간 밖의 행이 있습니다: {path}")
    return df.sort_values(["site_id", "ts"]).reset_index(drop=True)


def load_groups(sites_path: Path, composition_path: Path | None):
    if not sites_path.is_file():
        raise FileNotFoundError(
            f"원본 발전소 메타 파일이 없습니다: {sites_path}\n"
            "원본 데이터셋의 sites.csv를 프로젝트 data/ 폴더에 복사하세요."
        )
    sites = pd.read_csv(sites_path)
    needed = {"site_id", "capacity_kw"}
    if not needed.issubset(sites.columns):
        raise ValueError(f"sites.csv 필수 열 누락: {needed - set(sites.columns)}")
    if len(sites) != 253 or sites["site_id"].duplicated().any():
        raise ValueError("sites.csv에 고유한 발전소 253개가 있어야 합니다.")
    sites["site_id"] = sites["site_id"].astype("string")
    sites["capacity_kw"] = pd.to_numeric(sites["capacity_kw"], errors="raise")
    if sites["capacity_kw"].isna().any() or (sites["capacity_kw"] <= 0).any():
        raise ValueError("설비용량에 비정상값이 있습니다.")

    if composition_path is not None:
        if not composition_path.is_file():
            raise FileNotFoundError(f"지정한 그룹 배정 파일이 없습니다: {composition_path}")
        assign = pd.read_csv(composition_path, usecols=["site_id", "group"])
        source_label = f"제출 후보 배정: {composition_path.name}"
    else:
        if "current_group" not in sites.columns:
            raise ValueError("참조 그룹 열(current_group)이 sites.csv에 없습니다.")
        assign = sites[["site_id", "current_group"]].rename(
            columns={"current_group": "group"}
        ).copy()
        assign["group"] = assign["group"].map({"A": "G1", "B": "G2"})
        source_label = "기존 참조 배정 (A→G1, B→G2)"

    assign["site_id"] = assign["site_id"].astype("string")
    if len(assign) != 253 or assign["site_id"].duplicated().any():
        raise ValueError("그룹 배정은 253개 발전소를 중복 없이 정확히 포함해야 합니다.")
    if set(assign["site_id"]) != set(sites["site_id"]):
        raise ValueError("그룹 배정의 site_id 목록이 sites.csv와 일치하지 않습니다.")
    if set(assign["group"]) != {"G1", "G2"}:
        raise ValueError("group은 반드시 G1 또는 G2만 포함해야 합니다.")

    joined = assign.merge(
        sites[["site_id", "capacity_kw"]], on="site_id", validate="one_to_one"
    )
    caps = joined.groupby("group", observed=True)["capacity_kw"].sum().sort_index()
    if (caps < 20000).any():
        raise ValueError(f"그룹 최소 설비용량 20,000kW 미달: {caps.to_dict()}")
    return joined[["site_id", "group"]], caps, source_label


def load_price(train_path: Path, timestamps: pd.Series) -> pd.DataFrame:
    if not train_path.is_file():
        raise FileNotFoundError(f"전처리 학습 데이터가 없습니다: {train_path}")
    # SMP and k are the same for all sites in the same hour.
    price = pd.read_parquet(train_path, columns=["ts", "smp", "k"])
    price["ts"] = pd.to_datetime(price["ts"], utc=True, errors="raise")
    price = price[(price["ts"] >= DEV_START) & (price["ts"] < DEV_END)]
    price = price.drop_duplicates()
    if price.duplicated("ts").any():
        raise ValueError("한 시각에 서로 다른 SMP 또는 k가 존재합니다.")
    if price[["ts", "smp", "k"]].isna().any().any():
        raise ValueError("SMP 또는 시간 가중치 k에 결측이 있습니다.")
    if (price["smp"] < 0).any() or not price["k"].isin([1, 3]).all():
        raise ValueError("SMP가 음수이거나 k가 1/3이 아닙니다.")
    expected_k = np.where(price["ts"].dt.tz_convert("Asia/Seoul").dt.hour.between(11, 15), 3, 1)
    if not np.array_equal(price["k"].to_numpy(), expected_k):
        raise ValueError("k와 한국시간 11~15시 가중치 규칙이 일치하지 않습니다.")
    if set(price["ts"]) != set(timestamps):
        raise ValueError("SMP 시간축이 예측 파일의 검증 시간축과 일치하지 않습니다.")
    return price[["ts", "smp", "k"]]


def score_model(df: pd.DataFrame, assignment: pd.DataFrame, price: pd.DataFrame, name: str):
    x = df.merge(assignment, on="site_id", how="left", validate="many_to_one")
    if x["group"].isna().any():
        raise ValueError(f"{name}: 그룹이 누락된 발전소가 있습니다.")

    # First sum site errors per group, THEN take absolute value.
    x["error_kwh"] = x["pred_kwh"] - x["actual_kwh"]
    grouped = x.groupby(["ts", "group"], observed=True, as_index=False).agg(
        pred_kwh=("pred_kwh", "sum"),
        actual_kwh=("actual_kwh", "sum"),
        error_kwh=("error_kwh", "sum"),
        site_count=("site_id", "nunique"),
    )
    grouped = grouped.merge(price, on="ts", how="left", validate="many_to_one")
    if grouped[["smp", "k"]].isna().any().any():
        raise ValueError(f"{name}: 정산 가격이 누락됐습니다.")
    if len(grouped) != len(price) * 2:
        raise ValueError(f"{name}: 시각마다 G1·G2가 모두 존재하지 않습니다.")

    group_sites = assignment.groupby("group", observed=True)["site_id"].nunique()
    count_by_group = grouped.groupby("group", observed=True)["site_count"].unique()
    for g in ("G1", "G2"):
        if not np.array_equal(count_by_group[g], [group_sites[g]]):
            raise ValueError(f"{name}: {g} 일부 발전소·시각의 실적/예측이 누락됐습니다.")

    grouped["abs_error_kwh"] = grouped["error_kwh"].abs()
    grouped["penalty_krw"] = (
        0.10 * grouped["smp"] * grouped["k"] * grouped["abs_error_kwh"]
    )
    grouped["revenue_krw"] = grouped["smp"] * grouped["actual_kwh"]
    grouped["experiment"] = name
    grouped["kst_hour"] = grouped["ts"].dt.tz_convert("Asia/Seoul").dt.hour

    deviation = x["error_kwh"].to_numpy(dtype=np.float64)
    actual = x["actual_kwh"].to_numpy(dtype=np.float64)
    w = x.merge(price, on="ts", how="left", validate="many_to_one")
    weight = w["smp"].to_numpy(dtype=np.float64) * w["k"].to_numpy(dtype=np.float64)
    site_proxy = 0.10 * np.sum(weight * np.abs(deviation))

    revenue = float(grouped["revenue_krw"].sum())
    penalty = float(grouped["penalty_krw"].sum())
    peak = float(grouped.loc[grouped["kst_hour"].between(11, 15), "penalty_krw"].sum())
    metric = {
        "experiment": name,
        "rows": len(df),
        "mae_kwh": float(np.abs(deviation).mean()),
        "rmse_kwh": float(np.sqrt(np.mean(deviation**2))),
        "bias_kwh": float(deviation.mean()),
        "site_weighted_penalty_proxy_krw": float(site_proxy),
        "group_penalty_krw": penalty,
        "group_penalty_peak_11_15_krw": peak,
        "group_penalty_other_hours_krw": penalty - peak,
        "revenue_krw": revenue,
        "settlement_krw": revenue - penalty,
        "group_total_abs_error_kwh": float(grouped["abs_error_kwh"].sum()),
    }
    return metric, grouped


def main():
    parser = argparse.ArgumentParser(description="Compare OIBC EXP-01/02/03 on identical Dev data")
    parser.add_argument("--composition", type=Path, default=None,
                        help="Optional composition.csv (site_id,group). Default: sites.csv current_group A/B")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project root containing data/ and outputs/")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    composition = args.composition
    if composition is not None and not composition.is_absolute():
        composition = root / composition
    assignment, caps, source_label = load_groups(root / "data" / "sites.csv", composition)

    frames = {}
    for name, filename in EXPERIMENTS.items():
        frames[name] = read_error_csv(out / filename)
    first_name = next(iter(EXPERIMENTS))
    reference = frames[first_name][["site_id", "ts", "actual_kwh"]]

    for name, df in frames.items():
        if len(df) != len(reference):
            raise ValueError(f"{name}: 행 수가 다릅니다. 비교를 중단합니다.")
        if not df[["site_id", "ts"]].equals(reference[["site_id", "ts"]]):
            raise ValueError(f"{name}: (site_id, ts) 키가 다른 모델과 일치하지 않습니다.")
        if not np.allclose(
            df["actual_kwh"].to_numpy(dtype=np.float64),
            reference["actual_kwh"].to_numpy(dtype=np.float64),
            rtol=0, atol=1e-3,
        ):
            raise ValueError(f"{name}: 실제 발전량 값이 다른 모델과 다릅니다.")
        if set(df["site_id"]) != set(assignment["site_id"]):
            raise ValueError(f"{name}: 발전소 목록이 sites.csv와 일치하지 않습니다.")

    timestamps = reference["ts"].drop_duplicates()
    if len(reference) != len(timestamps) * len(assignment):
        raise ValueError("검증 시각마다 전체 발전소가 존재하지 않습니다. 그룹 실적 비교 불가.")
    price = load_price(root / "data" / "train_v0.1.parquet", timestamps)

    scored = []
    hourly = []
    for name, df in frames.items():
        metrics, group_ts = score_model(df, assignment, price, name)
        scored.append(metrics)
        hourly.append(group_ts)

    result = pd.DataFrame(scored)
    base_penalty = float(result.loc[result["experiment"] == first_name, "group_penalty_krw"].iloc[0])
    result["penalty_improvement_vs_exp01_pct"] = (
        100 * (base_penalty - result["group_penalty_krw"]) / base_penalty
        if base_penalty != 0 else np.nan
    )
    result = result.sort_values("group_penalty_krw", ascending=True).reset_index(drop=True)

    score_path = out / "exp01_exp02_exp03_group_scores_dev.csv"
    hourly_path = out / "exp01_exp02_exp03_group_hourly_dev.csv"
    meta_path = out / "exp01_exp02_exp03_group_scores_dev_info.json"
    result.to_csv(score_path, index=False, encoding="utf-8-sig")
    pd.concat(hourly, ignore_index=True).sort_values(["experiment", "ts", "group"]).to_csv(
        hourly_path, index=False, encoding="utf-8-sig"
    )
    meta_path.write_text(json.dumps({
        "period_kst": "2025-03-01 01:00 <= ts < 2025-05-01 01:00",
        "group_assignment": source_label,
        "group_capacity_kw": {str(k): float(v) for k, v in caps.items()},
        "rows_each_model": int(len(reference)),
        "hours": int(len(timestamps)),
        "formula": "Penalty = 0.10 * sum_t SMP(t)*k(t)*(abs(sum_G1(pred-actual)) + abs(sum_G2(pred-actual)))",
        "note": "Historical dev-period evaluation only; not the test leaderboard score.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n====== EXP-01/02/03 | 동일 조건 VPP 그룹 페널티 비교 ======")
    print("그룹 구성:", source_label)
    print("그룹별 용량:", {g: round(float(v), 2) for g, v in caps.items()})
    print(f"검증: 발전소 {len(assignment)}개 x {len(timestamps)}시간 = {len(reference):,}행")
    print("\n낮을수록 좋은 지표: MAE / RMSE / 그룹 정산 페널티")
    print(result[[
        "experiment", "mae_kwh", "rmse_kwh", "group_penalty_krw",
        "penalty_improvement_vs_exp01_pct", "settlement_krw"
    ]].to_string(index=False, float_format=lambda n: f"{n:,.4f}"))
    print("\n11~15시 페널티는 이미 k=3이 적용된 금액입니다.")
    print("제출용 테스트 점수 아님 · 비교 기준은 동일한 3~4월 개발 검증 기간")
    print("\n저장 파일:", score_path, hourly_path, meta_path, sep="\n - ")


if __name__ == "__main__":
    main()
