"""
사용법:
  python score_file.py 예측파일.csv                 → 기본 배정(A→G1, B→G2)으로 채점
  python score_file.py 예측파일.csv 배정파일.csv     → 지정한 배정으로 채점
예측 파일 열: site_id, ts, pred_kwh  (csv 또는 parquet)
배정 파일 열: site_id, group
"""
import sys
from pathlib import Path
import pandas as pd
from scoring.score import score_preds, check_composition

D = Path(__file__).resolve().parent.parent / "data" / "OIBC_2026_DATA" / "handover 2" / "01_participant_dataset"


def read(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

if len(sys.argv) < 2:
    print("사용법: python score_file.py 예측파일.csv [배정파일.csv]")
    sys.exit(1)

preds = read(sys.argv[1])
sites = pd.read_csv(D / "sites.csv")

if len(sys.argv) > 2:
    comp = read(sys.argv[2])
    problems = check_composition(comp, sites)
    if problems:
        print("배정 파일 문제:", problems)
        sys.exit(1)
else:
    comp = sites[["site_id"]].assign(group=sites["current_group"].map({"A": "G1", "B": "G2"}))

r = score_preds(preds, comp, D)
print(f"채점 기간: {r['period']}")
print(f"정산금 S : {r['S']:,.0f} 원")
print(f"벌금     : {r['penalty']:,.0f} 원")
print(f"가중치 없는 절대오차 합: {r['abs_err_sum']:,.0f} kWh")