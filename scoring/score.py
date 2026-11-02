"""
OIBC 대회식 정산 점수 계산 (팀 공통)
S = Σ SMP·실적 − 0.10 × Σ SMP·k·|예측 − 실적|   ← 그룹별·시각별 합계 기준
k: ts의 시(KST)가 11~15이면 3, 아니면 1
오차 정의: 예측 − 실제
"""
import numpy as np
import pandas as pd
from pathlib import Path

KST = "Asia/Seoul"


def to_kst(s: pd.Series) -> pd.Series:
    """시각 열을 KST(ns 단위)로 통일. 문자열·타임존 있는 시각 모두 처리."""
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert(KST).astype(f"datetime64[ns, {KST}]")


def load_smp(path) -> pd.DataFrame:
    smp = pd.read_csv(path)
    smp["ts"] = to_kst(smp["ts"])
    if "k" not in smp.columns:  # train/smp.csv에는 k가 없음 → 규칙으로 생성
        smp["k"] = np.where(smp["ts"].dt.hour.between(11, 15), 3, 1)
    return smp[["ts", "smp", "k"]]


def _group_table(site_preds, composition, smp):
    df = site_preds.merge(composition[["site_id", "group"]], on="site_id", how="left")
    if df["group"].isna().any():
        raise ValueError("composition에 없는 site_id가 있어요")
    g = df.groupby(["group", "ts"], as_index=False)[["actual_kwh", "pred_kwh"]].sum()
    g = g.merge(smp, on="ts", how="inner")
    g["abs_err"] = (g["pred_kwh"] - g["actual_kwh"]).abs()
    g["pen"] = 0.10 * g["smp"] * g["k"] * g["abs_err"]
    return g

def score(site_preds, composition, smp) -> dict:
    """
    site_preds : site_id, ts, actual_kwh, pred_kwh (실적이 있는 행만)
    composition: site_id, group
    smp        : ts, smp, k
    """
    g = _group_table(site_preds, composition, smp)
    revenue = (g["smp"] * g["actual_kwh"]).sum()
    penalty = g["pen"].sum()
    return {
        "S": revenue - penalty,
        "revenue": revenue,
        "penalty": penalty,
        "abs_err_sum": g["abs_err"].sum(),  # 동점 처리용(가중치 없는 절대오차 합)
        "hours": g["ts"].nunique(),
    }


def breakdown(site_preds, composition, smp):
    """벌금이 어느 달·어느 가중치 시간대에서 나왔는지 비율로."""
    g = _group_table(site_preds, composition, smp)
    total = g["pen"].sum()
    by_month = (g.groupby(g["ts"].dt.month)["pen"].sum() / total).round(3)
    by_k = (g.groupby("k")["pen"].sum() / total).round(3)
    return by_month, by_k


def check_composition(composition, sites) -> list:
    """제출 전 배정 검사. 문제 목록을 돌려줌(비어 있으면 통과)."""
    problems = []
    ids = set(sites["site_id"])
    c = composition
    if c["site_id"].duplicated().any():
        problems.append("중복된 site_id")
    if set(c["site_id"]) != ids:
        problems.append("빠지거나 없는 site_id")
    if not set(c["group"]) <= {"G1", "G2"}:
        problems.append("group은 G1/G2만 가능")
    cap = c.merge(sites[["site_id", "capacity_kw"]], on="site_id").groupby("group")["capacity_kw"].sum()
    for grp in ("G1", "G2"):
        if cap.get(grp, 0) < 20000:
            problems.append(f"{grp} 용량 {cap.get(grp, 0):,.1f} kW < 20,000")
    return problems

def load_actuals(data_dir, start=None, end=None) -> pd.DataFrame:
    """generation.parquet에서 실제 발전량을 읽어옴 (음수는 0으로)."""
    gen = pd.read_parquet(Path(data_dir) / "train" / "generation.parquet")
    gen["ts"] = to_kst(gen["ts"])
    if start is not None:
        gen = gen[gen["ts"] >= start]
    if end is not None:
        gen = gen[gen["ts"] <= end]
    gen["actual_kwh"] = gen["energy"].clip(lower=0)
    return gen[["site_id", "ts", "actual_kwh"]]

def score_preds(preds, composition, data_dir) -> dict:
    """
    예측 파일(site_id, ts, pred_kwh)만 받아서 실제값을 붙여 채점.
    예측 파일에 들어 있는 기간만 채점하고, 그 기간에 실적이 있는 행은 전부 예측이 있어야 함.
    """
    p = preds[["site_id", "ts", "pred_kwh"]].copy()
    p["ts"] = to_kst(p["ts"])

    # 예측 파일 검사
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("같은 (site_id, ts)가 두 번 이상 있어요")
    if (p["pred_kwh"] < 0).any() or not np.isfinite(p["pred_kwh"]).all():
        raise ValueError("pred_kwh에 음수·빈 값·무한대가 있어요")

    start, end = p["ts"].min(), p["ts"].max()
    act = load_actuals(data_dir, start, end)
    df = act.merge(p, on=["site_id", "ts"], how="left")
    missing = df["pred_kwh"].isna().sum()
    if missing:
        raise ValueError(f"실적은 있는데 예측이 없는 행이 {missing:,}개 있어요")

    smp = load_smp(Path(data_dir) / "train" / "smp.csv")
    result = score(df, composition, smp)
    result["period"] = f"{start} ~ {end}"
    return result

if __name__ == "__main__":
    # 작은 예제 검산: 13시(k=3), SMP 100원, 두 발전소 오차 +100 / −100
    ts = to_kst(pd.Series(["2024-07-01 13:00:00+09:00"] * 2))
    sp = pd.DataFrame({"site_id": ["P1", "P2"], "ts": ts,
                       "actual_kwh": [1000.0, 1000.0], "pred_kwh": [1100.0, 900.0]})
    smp = pd.DataFrame({"ts": ts[:1], "smp": [100.0], "k": [3]})

    same = score(sp, pd.DataFrame({"site_id": ["P1", "P2"], "group": ["G1", "G1"]}), smp)
    diff = score(sp, pd.DataFrame({"site_id": ["P1", "P2"], "group": ["G1", "G2"]}), smp)
    print("같은 그룹:", same["penalty"], same["S"])   # 기대: 0, 200000
    print("다른 그룹:", diff["penalty"], diff["S"])   # 기대: 6000, 194000
    assert same["penalty"] == 0 and same["S"] == 200000
    assert abs(diff["penalty"] - 6000) < 1e-9 and abs(diff["S"] - 194000) < 1e-9
    print("검산 통과 ✅")