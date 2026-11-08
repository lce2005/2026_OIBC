"""
제출 파일 생성·검사.

1) 생성+검사: python submit/make_submission.py --round 2 모델이름
   outputs/round2/preds/{모델이름}_test_predictions.csv  (평가 기간 발전소별 예측, 253개소 × 8,760시각)
   outputs/round2/dev_opt/composition_{모델이름}_dev.csv (group/search.py 최적화 배정)
   → outputs/round2/submissions/{모델이름}_OPT/{모델이름}_OPT_composition.csv, {모델이름}_OPT_forecast.csv, info.json
2) 검사만:   python submit/make_submission.py --check 폴더   (폴더 안 composition.csv, forecast.csv 검사)

검사 항목은 공식 Q&A 공지 "6. 제출 전 확인" + README 형식:
  253개소가 중복·누락 없이 G1/G2 하나씩 / 두 그룹 모두 20,000kW 이상
  17,520행 = G1·G2 × smp.csv의 8,760시각, ts에 시간대 포함 / forecast_kwh ≥ 0, 빈 값·무한대 없음
하나라도 실패하면 "제출 불가"로 끝남 (형식 오류도 제출 횟수에서 차감되므로).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import score as sc

KST = "Asia/Seoul"
problems = []


def bad(msg):
    problems.append(msg); print(f"  [실패] {msg}")


def ok(msg):
    print(f"  [통과] {msg}")


def check(folder: Path):
    sites = pd.read_csv(sc.DATA_DIR / "sites.csv")
    smp = pd.read_csv(sc.DATA_DIR / "smp.csv")
    smp_ts = sc.to_kst(pd.to_datetime(smp["ts"]))
    print(f"검사 대상: {folder}")

    # ---- composition.csv
    cps = sorted(folder.glob("*composition.csv"))
    if len(cps) != 1:
        bad(f"폴더에 composition 파일이 정확히 1개여야 함 (지금 {len(cps)}개)"); return finish(folder)
    cp = cps[0]
    c = pd.read_csv(cp)
    if list(c.columns) != ["site_id", "group"]:
        bad(f"composition 열이 site_id,group 이어야 함: {list(c.columns)}")
    c["site_id"] = c["site_id"].astype(str)
    if c["site_id"].duplicated().any(): bad("composition에 중복 site_id")
    if set(c["site_id"]) != set(sites["site_id"].astype(str)): bad("composition의 발전소가 sites.csv와 다름 (빠짐/없는 ID)")
    if len(c) != 253: bad(f"composition 행 수 {len(c)} ≠ 253")
    if not set(c["group"].astype(str)) <= {"G1", "G2"} or set(c["group"]) != {"G1", "G2"}: bad(f"group 값은 G1/G2 둘 다 있어야 함: {sorted(set(c['group'].astype(str)))}")
    cap = c.merge(sites[["site_id", "capacity_kw"]].assign(site_id=sites["site_id"].astype(str)), on="site_id", how="left").groupby("group")["capacity_kw"].sum()
    for g in ("G1", "G2"):
        if cap.get(g, 0) < 20000: bad(f"{g} 용량 {cap.get(g, 0):,.2f}kW < 20,000")
    if not problems: ok(f"composition: 253개소, G1 {cap.get('G1', 0):,.2f}kW ({(c.group == 'G1').sum()}개) / G2 {cap.get('G2', 0):,.2f}kW ({(c.group == 'G2').sum()}개)")
    n_before = len(problems)

    # ---- forecast.csv
    fps = sorted(folder.glob("*forecast.csv"))
    if len(fps) != 1:
        bad(f"폴더에 forecast 파일이 정확히 1개여야 함 (지금 {len(fps)}개)"); return finish(folder)
    fp = fps[0]
    f = pd.read_csv(fp)
    if list(f.columns) != ["group", "ts", "forecast_kwh"]:
        bad(f"forecast 열이 group,ts,forecast_kwh 이어야 함: {list(f.columns)}")
    if len(f) != 17520: bad(f"forecast 행 수 {len(f):,} ≠ 17,520")
    raw_ts = f["ts"].astype(str)
    if not raw_ts.str.contains(r"[+-]\d\d:\d\d$").all(): bad("ts에 시간대(+09:00)가 없는 행이 있음")
    try:
        f["ts"] = sc.to_kst(pd.to_datetime(f["ts"], utc=True))
    except Exception as e:
        bad(f"ts를 시각으로 읽을 수 없음: {e}"); return finish(folder)
    if f.duplicated(["group", "ts"]).any(): bad("forecast에 (group, ts) 중복")
    for g in ("G1", "G2"):
        t = f.loc[f.group == g, "ts"]
        if len(t) != 8760 or set(t) != set(smp_ts): bad(f"{g} 시각이 smp.csv의 8,760시각과 다름 (행 {len(t):,})")
    v = pd.to_numeric(f["forecast_kwh"], errors="coerce")
    if v.isna().any(): bad(f"forecast_kwh 빈 값/숫자 아님 {int(v.isna().sum())}행")
    if not np.isfinite(v.dropna()).all(): bad("forecast_kwh에 무한대")
    if (v < 0).any(): bad(f"forecast_kwh 음수 {int((v < 0).sum())}행")
    if len(problems) == n_before:
        ok(f"forecast: 17,520행, 시각 일치, 값 범위 {v.min():,.1f} ~ {v.max():,.1f} kWh, 합계 G1 {v[f.group == 'G1'].sum() / 1e6:,.2f} / G2 {v[f.group == 'G2'].sum() / 1e6:,.2f} GWh")
    return finish(folder)


def finish(folder):
    if problems:
        print(f"\n제출 불가: 문제 {len(problems)}개. 고친 뒤 다시 검사하세요.")
        return False
    print("\n제출 가능: 모든 검사 통과")
    return True


def make(preds_path: Path, comp_path: Path, out: Path):
    sites = pd.read_csv(sc.DATA_DIR / "sites.csv")
    smp = pd.read_csv(sc.DATA_DIR / "smp.csv")
    smp_ts = sc.to_kst(pd.to_datetime(smp["ts"]))
    p = pd.read_parquet(preds_path) if preds_path.suffix == ".parquet" else pd.read_csv(preds_path)
    p = p[["site_id", "ts", "pred_kwh"]].copy()
    p["site_id"] = p["site_id"].astype(str)
    p["ts"] = sc.to_kst(pd.to_datetime(p["ts"], utc=True))
    if p.duplicated(["site_id", "ts"]).any(): raise ValueError("예측에 (site_id, ts) 중복")
    if set(p.site_id) != set(sites.site_id.astype(str)): raise ValueError("예측의 발전소가 253개소와 다름")
    if set(p.ts) != set(smp_ts) or len(p) != 253 * 8760: raise ValueError(f"예측이 253 × 8,760 = 2,216,280행이어야 함 (지금 {len(p):,}행)")
    if p.pred_kwh.isna().any() or not np.isfinite(p.pred_kwh).all(): raise ValueError("예측에 빈 값/무한대")
    neg = int((p.pred_kwh < 0).sum())
    if neg: print(f"  음수 예측 {neg}행 → 0으로 올림"); p["pred_kwh"] = p.pred_kwh.clip(lower=0)

    comp = pd.read_csv(comp_path)[["site_id", "group"]]
    comp["site_id"] = comp["site_id"].astype(str)
    problems_c = sc.check_composition(comp, sites.assign(site_id=sites.site_id.astype(str)))
    if problems_c: raise ValueError(f"배정 규칙 위반: {problems_c}")

    f = (p.merge(comp, on="site_id", how="left").groupby(["group", "ts"], as_index=False).pred_kwh.sum()
           .rename(columns={"pred_kwh": "forecast_kwh"}).sort_values(["group", "ts"]))
    f["ts"] = f["ts"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"(\d\d)(\d\d)$", r"\1:\2", regex=True)

    out.mkdir(parents=True, exist_ok=True)
    # 파일 이름에 폴더 이름을 붙여 제출 내역에서 구분되게 함 (2회차부터)
    comp.sort_values("site_id").to_csv(out / f"{out.name}_composition.csv", index=False)
    f.to_csv(out / f"{out.name}_forecast.csv", index=False)
    (out / "info.json").write_text(json.dumps({
        "preds": str(preds_path.relative_to(ROOT)), "composition": str(comp_path.relative_to(ROOT)),
        "made_at": pd.Timestamp.now(tz=KST).isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"생성: {out}/{out.name}_composition.csv, {out.name}_forecast.csv\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, help="제출 회차 (outputs/round{N}/)")
    ap.add_argument("model", nargs="?", help="모델 이름")
    ap.add_argument("--check", type=Path, help="검사만: composition.csv, forecast.csv 가 있는 폴더")
    a = ap.parse_args()
    if a.check:
        folder = a.check if a.check.is_absolute() else ROOT / a.check
    elif a.round and a.model:
        rdir = ROOT / "outputs" / f"round{a.round}"
        folder = make(rdir / "preds" / f"{a.model}_test_predictions.csv",
                      rdir / "dev_opt" / f"composition_{a.model}_dev.csv",
                      rdir / "submissions" / f"{a.model}_OPT")
    else:
        ap.error("--round N 모델이름  또는  --check 폴더 가 필요해요")
    sys.exit(0 if check(folder) else 1)
