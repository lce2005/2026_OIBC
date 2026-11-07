"""
모델 담당이 준 예측 파일(site_id, ts, pred_kwh)에 공통 정답을 붙여 오차표를 만들고, 공식 점수를 낸다.

실행 (2026_OIBC 폴더에서):
  python scoring/make_errors.py data/preds/모델_dev_predictions.csv [또 다른 예측 ...]
저장:
  outputs/{모델이름}_dev_errors.csv   (site_id, ts, actual_kwh, pred_kwh, error_kwh)  ← 팀 전달 형식, 그룹 탐색 입력
  outputs/model_scores_dev.csv       (모델별 공식 점수 표. 실행할 때마다 갱신)
정답은 score.py와 같은 곳(원본 generation.parquet, 음수는 0)에서 가져오므로 모든 모델이 같은 정답으로 비교된다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import score as sc

OUT = ROOT / "outputs"


def make(pred_path: Path):
    name = pred_path.stem.replace("_dev_predictions", "").replace("_predictions", "")
    p = pd.read_parquet(pred_path) if pred_path.suffix == ".parquet" else pd.read_csv(pred_path)
    p = p[["site_id", "ts", "pred_kwh"]].copy()
    p["site_id"] = p["site_id"].astype(str)
    p["ts"] = sc.to_kst(pd.to_datetime(p["ts"], utc=True))
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError(f"{name}: (site_id, ts) 중복")
    if p["pred_kwh"].isna().any() or not np.isfinite(p["pred_kwh"]).all():
        raise ValueError(f"{name}: 예측에 빈 값/무한대")
    neg = int((p["pred_kwh"] < 0).sum())
    if neg:
        print(f"  {name}: 음수 예측 {neg}행 → 0으로 올림")
        p["pred_kwh"] = p["pred_kwh"].clip(lower=0)

    actual = sc.load_actuals(sc.DATA_DIR, p["ts"].min(), p["ts"].max())
    actual["site_id"] = actual["site_id"].astype(str)
    e = actual.merge(p, on=["site_id", "ts"], how="left", validate="one_to_one")
    if e["pred_kwh"].isna().any():
        raise ValueError(f"{name}: 실적은 있는데 예측이 없는 행 {int(e['pred_kwh'].isna().sum()):,}개")
    e["error_kwh"] = e["pred_kwh"] - e["actual_kwh"]
    e = e.sort_values(["site_id", "ts"])
    e[["site_id", "ts", "actual_kwh", "pred_kwh", "error_kwh"]].to_csv(OUT / f"{name}_dev_errors.csv", index=False)

    sites = pd.read_csv(sc.DATA_DIR / "sites.csv")
    smp = sc.load_smp(sc.DATA_DIR / "train" / "smp.csv")
    s = sc.score(e, sc.default_composition(sites).assign(site_id=lambda d: d.site_id.astype(str)), smp)
    row = {"model": name, "rows": len(e), "hours": s["hours"],
           "mae_kwh": float(e["error_kwh"].abs().mean()),
           "rmse_kwh": float(np.sqrt((e["error_kwh"] ** 2).mean())),
           "bias_kwh": float(e["error_kwh"].mean()),
           "penalty_AB": s["penalty"], "S_AB": s["S"],
           "period": f"{e.ts.min()} ~ {e.ts.max()}", "pred_file": str(pred_path.relative_to(ROOT)) if pred_path.is_relative_to(ROOT) else str(pred_path)}
    print(f"  {name}: MAE {row['mae_kwh']:.4f} / 기본 배정 페널티 {s['penalty']:,.0f}원 → {OUT.name}/{name}_dev_errors.csv")
    return row


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    rows = []
    for a in sys.argv[1:]:
        path = Path(a) if Path(a).is_absolute() else ROOT / a
        rows.append(make(path))
    table_path = OUT / "model_scores_dev.csv"
    table = pd.DataFrame(rows)
    if table_path.exists():
        old = pd.read_csv(table_path)
        table = pd.concat([old[~old.model.isin(table.model)], table], ignore_index=True)
    table = table.sort_values("penalty_AB").reset_index(drop=True)
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print("\n" + table[["model", "mae_kwh", "rmse_kwh", "penalty_AB", "S_AB"]].to_string(index=False, float_format=lambda x: f"{x:,.4f}" if abs(x) < 100 else f"{x:,.0f}"))
