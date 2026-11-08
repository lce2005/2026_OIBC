"""
5월(holdout) 확인: 3~4월(dev) 오차로 찾은 배정이 안 본 달인 5월에서도 기존 A/B보다 나은지 채점한다.
배정은 바꾸지 않고 채점만 한다 (5월은 배정 탐색에 쓰지 않음, 일정 계획 4번).

실행 (2026_OIBC 폴더에서):
  python group/check_holdout.py --round 2 모델이름 [모델이름 ...]
필요 파일 (outputs/round{N}/):
  preds/{모델이름}_holdout_predictions.csv      (site_id, ts, pred_kwh, 5월 253개소 × 744시간)
  dev_opt/composition_{모델이름}_dev.csv         (group/search.py가 만든 최적화 배정)
저장:
  outputs/round{N}/holdout/holdout_check.csv  (모델별 5월 기존 A/B 벌금, 최적화 배정 벌금, 감소율. 실행할 때마다 갱신)
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import score as sc

HOLDOUT = (pd.Timestamp("2025-05-01 01:00", tz="Asia/Seoul"), pd.Timestamp("2025-06-01 00:00", tz="Asia/Seoul"))


def check(model: str, rdir: Path, sites, smp, ab):
    p = pd.read_csv(rdir / "preds" / f"{model}_holdout_predictions.csv")
    p["site_id"] = p["site_id"].astype(str)
    p["ts"] = sc.to_kst(pd.to_datetime(p["ts"], utc=True))
    if p["ts"].min() != HOLDOUT[0] or p["ts"].max() != HOLDOUT[1] or len(p) != 253 * 744:
        raise ValueError(f"{model}: 5월 예측이 253개소 × 744시간(2025-05-01 01:00 ~ 06-01 00:00)이어야 함 (지금 {len(p):,}행)")
    if p.duplicated(["site_id", "ts"]).any() or p["pred_kwh"].isna().any():
        raise ValueError(f"{model}: 중복 또는 빈 값")
    p["pred_kwh"] = p["pred_kwh"].clip(lower=0)

    actual = sc.load_actuals(sc.DATA_DIR, *HOLDOUT)
    actual["site_id"] = actual["site_id"].astype(str)
    e = actual.merge(p, on=["site_id", "ts"], how="left", validate="one_to_one")
    if e["pred_kwh"].isna().any():
        raise ValueError(f"{model}: 실적은 있는데 예측이 없는 행 {int(e['pred_kwh'].isna().sum()):,}개")

    comp = pd.read_csv(rdir / "dev_opt" / f"composition_{model}_dev.csv")
    comp["site_id"] = comp["site_id"].astype(str)
    problems = sc.check_composition(comp, sites)
    if problems:
        raise ValueError(f"{model}: 배정 규칙 위반 {problems}")

    pen_ab = sc.score(e, ab, smp)["penalty"]
    pen_opt = sc.score(e, comp, smp)["penalty"]
    err = (e["pred_kwh"] - e["actual_kwh"]).groupby(e["ts"]).sum()
    w = smp.set_index("ts").reindex(err.index)
    lower = float(0.10 * (w["smp"] * w["k"] * err.abs()).sum())   # 한 그룹에 다 넣었을 때 = 더 못 줄이는 선
    return {"model": model, "holdout_penalty_AB": pen_ab, "holdout_penalty_opt": pen_opt,
            "holdout_lower_bound": lower, "reduction_%": 100 * (1 - pen_opt / pen_ab),
            "holdout_mae_kwh": float((e["pred_kwh"] - e["actual_kwh"]).abs().mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    rdir = ROOT / "outputs" / f"round{a.round}"
    sites = pd.read_csv(sc.DATA_DIR / "sites.csv"); sites["site_id"] = sites["site_id"].astype(str)
    smp = sc.load_smp(sc.DATA_DIR / "train" / "smp.csv")
    ab = sc.default_composition(sites)
    rows = [check(m, rdir, sites, smp, ab) for m in a.models]
    out = rdir / "holdout" / "holdout_check.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        table = pd.concat([old[~old.model.isin(table.model)], table], ignore_index=True)
    table = table.sort_values("holdout_penalty_opt").reset_index(drop=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{'모델':52s} {'5월 기존A/B':>12s} {'5월 최적화':>12s} {'감소':>6s} {'더 못줄이는 선':>13s} {'MAE':>7s}")
    for r in table.itertuples():
        print(f"{r.model:52s} {r.holdout_penalty_AB:>12,.0f} {r.holdout_penalty_opt:>12,.0f} {r._5:>5.1f}% {r.holdout_lower_bound:>13,.0f} {r.holdout_mae_kwh:>7.3f}")
    print(f"\n저장: {out.relative_to(ROOT)}")
