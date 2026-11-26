"""
야간 양수 점검: 태양 고도를 ts 정각 기준 vs 구간 중앙(ts − 30분) 기준으로 계산해 비교
실행: 레포 최상위 폴더에서  python eda/night_check.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "OIBC_2026_DATA" / "handover 2" / "01_participant_dataset"


def solar_elevation(ts, lat, lon):
    """주어진 시각의 태양 고도(도), NOAA 근사식."""
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


sites = pd.read_csv(D / "sites.csv")
gen = pd.read_parquet(D / "train" / "generation.parquet")
gen = gen.merge(sites[["site_id", "lat", "lon"]], on="site_id")
lat, lon = gen["lat"].to_numpy(), gen["lon"].to_numpy()

gen["elev_end"] = solar_elevation(gen["ts"], lat, lon)                              # 정각 기준
gen["elev_mid"] = solar_elevation(gen["ts"] - pd.Timedelta(minutes=30), lat, lon)   # 구간 중앙 기준

pos = gen["energy"] > 0
total_energy = gen.loc[pos, "energy"].sum()

print(f"전체 행: {len(gen):,} / 발전량 > 0 행: {pos.sum():,}\n")
print(f"{'기준':<16}{'밤 행':>12}{'밤인데 발전량>0':>18}{'깊은 밤(≤−18°) 양수':>22}{'밤 양수 발전량 비중':>20}")
for name, col in [("정각(ts)", "elev_end"), ("중앙(ts−30분)", "elev_mid")]:
    night = gen[col] <= 0
    night_pos = night & pos
    deep_pos = (gen[col] <= -18) & pos
    share = gen.loc[night_pos, "energy"].sum() / total_energy
    print(f"{name:<16}{night.sum():>12,}{night_pos.sum():>18,}{deep_pos.sum():>22,}{share:>20.3%}")

# 중앙 기준에서도 남는 밤 양수가 어느 시간대에 있는지
mid_np = gen[(gen["elev_mid"] <= 0) & pos]
print("\n[중앙 기준에서도 남는 밤 양수 - 시간대(ts 시)별 행 수]")
print(mid_np["ts"].dt.hour.value_counts().sort_index().to_string())

# 깊은 밤 양수가 특정 발전소에 몰려 있는지
deep = gen[(gen["elev_mid"] <= -18) & pos]
print(f"\n[깊은 밤 양수가 있는 발전소 수: {deep['site_id'].nunique()}개, 상위 5개]")
print(deep["site_id"].value_counts().head().to_string())