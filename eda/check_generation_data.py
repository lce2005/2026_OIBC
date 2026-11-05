"""
팀 공통 데이터(주영님 train_vX.parquet)의 발전량 처리가 합의대로인지, 대회 원본에서 다시 계산해 확인한다.
합의 근거: 문서 1 「발전량 점검 및 처리 방침」, 11/3 합의(use_train), 11/4 확정 기간

실행 (2026_OIBC 폴더에서):
  python eda/check_generation_data.py data/data_v0.4/train_v0.4.parquet
새 데이터 버전이 나오면 경로만 바꿔 다시 실행. 하나라도 실패하면 마지막에 실패 목록이 나옴.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import score as sc                       # 공통 점수 코드 (정답 비교용)
sys.path.insert(0, str(sc.DATA_DIR))
import baseline as official              # 대회 제공 baseline.py의 solar() (태양 고도, 구간 중앙 기준)

KST = "Asia/Seoul"
T = lambda s: pd.Timestamp(s, tz=KST)
SPLIT = {   # 11/4 확정 기간
    "train":   (T("2023-06-01 01:00"), T("2025-03-01 00:00")),
    "dev":     (T("2025-03-01 01:00"), T("2025-05-01 00:00")),
    "holdout": (T("2025-05-01 01:00"), T("2025-06-01 00:00")),
}
TARGET_COLS = ["energy_kwh", "energy_raw_kwh", "cf_target", "is_suspect_zero", "use_train"]
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'통과' if ok else '실패'}] {name}  {detail}", flush=True)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/data_v0.4/train_v0.4.parquet"
    path = path if path.is_absolute() else ROOT / path
    print(f"확인 대상: {path.name}\n")

    d = pd.read_parquet(path, columns=["site_id", "ts", "split", "energy_kwh", "energy_raw_kwh", "cf_target",
                                       "is_suspect_zero", "is_daylight", "use_train", "elev"])
    gen = pd.read_parquet(sc.DATA_DIR / "train" / "generation.parquet")
    sites = pd.read_csv(sc.DATA_DIR / "sites.csv")
    for x in (d, gen):
        x["site_id"] = x["site_id"].astype(str)
        x["ts"] = sc.to_kst(pd.to_datetime(x["ts"]))
        x.sort_values(["site_id", "ts"], inplace=True)
        x.reset_index(drop=True, inplace=True)

    # 1. 행: 기록 있는 시각만 (1번 A안)
    same_keys = len(d) == len(gen) and d[["site_id", "ts"]].equals(gen[["site_id", "ts"]])
    check("행 = 원본 발전량 행 그대로 (기록 전 시각은 행 없음, 1번 A안)", same_keys and not d.duplicated(["site_id", "ts"]).any(),
          f"{len(d):,}행 / 원본 {len(gen):,}행, 발전소 {d.site_id.nunique()}개")
    if not same_keys:
        print("행이 원본과 달라 이후 비교를 멈춤"); return finish()

    raw = gen["energy"].to_numpy(float)
    # 2. 음수 (2번 A안)
    check("원본 보존: energy_raw_kwh = 원본 발전량", np.allclose(d.energy_raw_kwh.to_numpy(float), raw, rtol=0, atol=1e-4, equal_nan=True),
          f"원본 음수 {int((raw < 0).sum())}건")
    check("음수는 0으로: energy_kwh = 원본을 0 아래로 못 내려가게", np.allclose(d.energy_kwh.to_numpy(float), np.clip(raw, 0, None), rtol=0, atol=1e-4),
          f"energy_kwh 음수 {int((d.energy_kwh < 0).sum())}건")
    act = sc.load_actuals(sc.DATA_DIR).sort_values(["site_id", "ts"]).reset_index(drop=True)
    check("점수 코드(score.py) 정답 = energy_kwh", np.allclose(act.actual_kwh.to_numpy(float), d.energy_kwh.to_numpy(float), rtol=0, atol=1e-4))

    # 3. 이용률 (4번 C안, 11번)
    cap = d.site_id.map(sites.set_index("site_id").capacity_kw).to_numpy(float)
    cf = np.minimum(np.clip(raw, 0, None) / cap, 1.2)
    check("cf_target = energy_kwh ÷ 용량, 상한 1.2 (4번 C안, 11번)", np.allclose(d.cf_target.to_numpy(float), cf, rtol=1e-5, atol=1e-6),
          f"최대 {float(d.cf_target.max()):.3f}")

    # 4. 하루 종일 0 (3번 판정 기준): (발전소, day_id) 20행 이상인데 합 ≤ 0
    day = (d.ts - pd.Timedelta(hours=1)).dt.floor("D")
    agg = pd.DataFrame({"site_id": d.site_id, "day": day, "e": raw}).groupby(["site_id", "day"]).e.agg(["size", "sum"])
    bad = agg[(agg["size"] >= 20) & (agg["sum"] <= 0)]
    flag = pd.MultiIndex.from_arrays([d.site_id, day]).isin(bad.index)
    check("is_suspect_zero = 20행 이상 기록 + 하루 합 0 (3번 기준)", np.array_equal(flag, d.is_suspect_zero.to_numpy() == 1),
          f"{len(bad)}개 발전소·일, {int(flag.sum()):,}행")

    # 5. 야간 판정: 구간 중앙(ts−30분) 기준 태양 고도 > 0
    loc = sites.set_index("site_id")
    elev, _, _ = official.solar(pd.DatetimeIndex(d.ts), d.site_id.map(loc.lat).to_numpy(float), d.site_id.map(loc.lon).to_numpy(float))
    diff = np.abs(elev.astype(float) - d.elev.to_numpy(float))
    mism = int(((elev > 0) != (d.is_daylight.to_numpy() == 1)).sum())
    check("elev = 구간 중앙 기준 태양 고도 (baseline.py solar와 같음)", diff.max() < 0.01, f"최대 차이 {diff.max():.4f}°")
    check("is_daylight = 태양 고도 > 0", mism == 0, f"불일치 {mism}행, 야간 {int((d.is_daylight == 0).sum()):,}행")

    # 6. 학습 행 (11/3 합의: 3번 D안 + 4번 A안)
    rule = (d.is_daylight == 1) & (d.is_suspect_zero == 0)
    check("use_train = 낮 & 하루 종일 0 아님 (11/3 합의)", np.array_equal(rule.to_numpy(), d.use_train.to_numpy() == 1),
          f"학습 행 {int(rule.sum()):,}")

    # 7. 구간 (11/4 확정 기간)
    for name, (s, e) in SPLIT.items():
        p = d[d.split == name]
        hours = len(pd.date_range(s, e, freq="h"))
        full = "" if name == "train" else f", 253개소×{hours}시간 꽉 참 {len(p) == 253 * hours and p.site_id.nunique() == 253}"
        ok = len(p) > 0 and p.ts.min() == s and p.ts.max() == e and (name == "train" or len(p) == 253 * hours)
        check(f"split={name} 경계 = 확정 기간", ok, f"{p.ts.min()} ~ {p.ts.max()}, {len(p):,}행{full}")
    others = sorted(set(d.split.astype(str)) - set(SPLIT) - {"pre"})
    check("그 밖의 구간 없음 (pre는 학습 시작 이전)", not others and d[d.split == "pre"].ts.max() < SPLIT["train"][0], f"기타 {others}")

    # 8. 평가 파일에 정답 열 없음
    test = path.with_name(path.name.replace("train_", "test_"))
    if test.exists():
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(test).schema_arrow.names
        rows = pq.ParquetFile(test).metadata.num_rows
        leak = [c for c in TARGET_COLS if c in cols]
        check("평가 파일에 정답·학습용 열 없음", not leak and rows == 253 * 8760, f"{test.name}: {rows:,}행, 남은 정답 열 {leak}")
    finish()


def finish():
    fails = [n for n, ok in results if not ok]
    print(f"\n총 {len(results)}개 중 통과 {len(results) - len(fails)}개" + (f", 실패: {fails}" if fails else ""))


if __name__ == "__main__":
    main()
