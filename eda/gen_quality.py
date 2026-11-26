"""
발전량 점검: 발전소별 기록 기간, 누락, 음수, 낮 0, 하루 종일 0, 최대 이용률
실행: 레포 최상위 폴더에서  python eda/gen_quality.py
결과: outputs/gen_quality.csv (로컬 전용, 깃에 올리지 않음)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "OIBC_2026_DATA" / "handover 2" / "01_participant_dataset"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

sites = pd.read_csv(D / "sites.csv")
gen = pd.read_parquet(D / "train" / "generation.parquet")
gen = gen.sort_values(["site_id", "ts"]).reset_index(drop=True)
gen["hour"] = gen["ts"].dt.hour
gen["date"] = gen["ts"].dt.date
gen["neg"] = gen["energy"] < 0
gen["is_day"] = gen["hour"].between(11, 15)
gen["day_zero"] = gen["is_day"] & (gen["energy"] == 0)
gen["gap_h"] = gen.groupby("site_id")["ts"].diff().dt.total_seconds() / 3600

g = gen.groupby("site_id")
q = pd.DataFrame({
    "start": g["ts"].min(),
    "end": g["ts"].max(),
    "n_rows": g.size(),
    "neg_rows": g["neg"].sum(),
    "day_rows": g["is_day"].sum(),
    "day_zero_rows": g["day_zero"].sum(),
    "max_gap_h": g["gap_h"].max(),
    "max_energy": g["energy"].max(),
})
q["expected_rows"] = ((q["end"] - q["start"]).dt.total_seconds() / 3600 + 1).astype(int)
q["missing_ratio"] = 1 - q["n_rows"] / q["expected_rows"]
q["day_zero_ratio"] = q["day_zero_rows"] / q["day_rows"]

daily = gen.groupby(["site_id", "date"])["energy"].sum()
q["days"] = daily.groupby("site_id").size()
q["zero_days"] = (daily == 0).groupby("site_id").sum()

q = q.join(sites.set_index("site_id")[["capacity_kw", "current_group"]])
q["max_cf"] = q["max_energy"] / q["capacity_kw"]
q = q.drop(columns=["day_rows", "max_energy"])
num_cols = q.select_dtypes("number").columns
q[num_cols] = q[num_cols].round(4)
q.to_csv(OUT / "gen_quality.csv", encoding="utf-8-sig")

# ---- 요약 출력 ----
print(f"발전소 수: {len(q)}")
print(f"\n[기록 시작 연도별]\n{q['start'].dt.year.value_counts().sort_index().to_string()}")
print(f"\n빠진 비율 10% 넘는 발전소: {(q['missing_ratio'] > 0.1).sum()}개")
print(f"최장 연속 공백: 최대 {q['max_gap_h'].max():.0f}시간, 중앙값 {q['max_gap_h'].median():.0f}시간")
neg = gen[gen["neg"]]
print(f"\n음수 있는 발전소: {(q['neg_rows'] > 0).sum()}개 (총 {int(q['neg_rows'].sum())}행)")
print(f"음수 값 범위: {neg['energy'].min():.2f} ~ {neg['energy'].max():.2f}")
print(f"음수 발생 시간대(ts 시):\n{neg['hour'].value_counts().sort_index().to_string()}")
print(f"\n낮 0 비율 5% 넘는 발전소: {(q['day_zero_ratio'] > 0.05).sum()}개")
print(f"하루 전체 0인 날 10일 넘는 발전소: {(q['zero_days'] > 10).sum()}개")
print(f"최대 이용률 범위: {q['max_cf'].min():.2f} ~ {q['max_cf'].max():.2f}")
print(f"\n저장: {OUT / 'gen_quality.csv'}")