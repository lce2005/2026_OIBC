"""
특정 발전소(기본 PV240)의 깊은 밤 양수 원인 점검
- 시간대별 평균 이용률을 다른 발전소와 비교
- 깊은 밤 양수가 언제·몇 시에 생겼는지
- 시각이 몇 시간 어긋나 있는지 추정 (전체 기간, 월별)
실행: 레포 최상위 폴더에서  python eda/pv240_check.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "OIBC_2026_DATA" / "handover 2" / "01_participant_dataset"
SITE = "PV240"


def solar_elevation(ts, lat, lon):
    t = ts.dt.tz_convert("UTC")
    doy = t.dt.day_of_year.to_numpy()
    hr = (t.dt.hour + t.dt.minute / 60).to_numpy()
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return 90 - np.degrees(np.arccos(np.clip(cz, -1, 1)))


def lag_corr(site_cf, fleet_cf, lags=range(-12, 13)):
    """발전소 시각을 lag시간 밀었을 때 전체 평균 패턴과의 상관."""
    out = {}
    for lag in lags:
        s = site_cf.copy()
        s.index = s.index + pd.Timedelta(hours=lag)
        j = pd.concat([s.rename("a"), fleet_cf.rename("b")], axis=1, join="inner")
        out[lag] = j["a"].corr(j["b"])
    return pd.Series(out)


sites = pd.read_csv(D / "sites.csv")
gen = pd.read_parquet(D / "train" / "generation.parquet")
gen = gen.merge(sites[["site_id", "capacity_kw", "lat", "lon"]], on="site_id")
gen["cf"] = gen["energy"].clip(lower=0) / gen["capacity_kw"]
gen["hour"] = gen["ts"].dt.hour
gen["elev"] = solar_elevation(gen["ts"] - pd.Timedelta(minutes=30),
                              gen["lat"].to_numpy(), gen["lon"].to_numpy())

me = gen[gen["site_id"] == SITE]
others = gen[gen["site_id"] != SITE]
print(f"[{SITE}] 용량 {me['capacity_kw'].iloc[0]:.1f} kW, 기록 {len(me):,}행, "
      f"{me['ts'].min()} ~ {me['ts'].max()}")

# 1) 시간대별 평균 이용률 비교
prof = pd.DataFrame({
    SITE: me.groupby("hour")["cf"].mean(),
    "다른 발전소 평균": others.groupby("hour")["cf"].mean(),
}) * 100
print("\n[1] 시간대별(ts 시) 평균 이용률(%)")
print(prof.round(1).to_string())
print(f"최대 시각: {SITE} = {prof[SITE].idxmax()}시, 다른 발전소 = {prof['다른 발전소 평균'].idxmax()}시")

# 2) 깊은 밤 양수가 언제 생겼나
deep = me[(me["elev"] <= -18) & (me["energy"] > 0)]
print(f"\n[2] 깊은 밤(태양 고도 ≤ −18°) 양수: {len(deep):,}행")
if len(deep):
    print(f"기간: {deep['ts'].min()} ~ {deep['ts'].max()}")
    print(f"이용률 평균 {deep['cf'].mean():.3f}, 최대 {deep['cf'].max():.3f}")
    print("월별 행 수:")
    print(deep["ts"].dt.strftime("%Y-%m").value_counts().sort_index().to_string())

# 3) 시각 어긋남 추정
fleet = others.groupby("ts")["cf"].mean()
site_cf = me.set_index("ts")["cf"]
c = lag_corr(site_cf, fleet)
print(f"\n[3] 시각 어긋남 추정 (전체 기간)")
print(f"그대로(0시간) 상관 {c[0]:.3f} / 가장 잘 맞는 이동 {c.idxmax():+d}시간 (상관 {c.max():.3f})")

print("\n월별로 가장 잘 맞는 이동 (0이 아니면 그달 시각이 어긋난 것)")
rows = []
for month, grp in me.groupby(me["ts"].dt.strftime("%Y-%m")):
    cm = lag_corr(grp.set_index("ts")["cf"], fleet)
    if cm.isna().all() or len(grp) < 48:   # 기록이 너무 적은 달은 건너뜀
        continue
    rows.append((month, int(cm.idxmax()), round(cm.max(), 3), round(cm[0], 3)))
print(pd.DataFrame(rows, columns=["월", "최적 이동(시간)", "그때 상관", "그대로 상관"]).to_string(index=False))