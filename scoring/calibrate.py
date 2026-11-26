"""
예측 수준 보정(배율) 점검: 모델 예측이 실제보다 일관되게 많은지(적은지) 보고, 배율을 곱했을 때 벌금이 주는지 5월로 검증한다.

원리
  벌금은 |예측-실제| 기준이라, 모델이 늘 조금 높게(낮게) 예측하면 예측 전체에 배율 하나를 곱해 낮추(높이)기만 해도 벌금이 준다.
  배율 = 실제 발전량 합계 ÷ 예측 발전량 합계 (1.0이면 딱 맞음, 1보다 작으면 예측이 많았던 것).

검증 방식 (5월은 배율을 정하는 데 쓰지 않음)
  ① 3~4월(dev)로 배율을 정한다 → 그 배율을 5월(holdout) 예측에 곱해 5월 벌금이 주는지 본다.
  ② 같은 배율을 3~5월 14주에 주별로 곱해, 벌금이 준 주가 2/3(10주) 이상인지 본다 (한두 주에만 맞는 배율 걸러내기).
  ①②를 모두 만족하면 '검증 통과'. 통과한 모델만 제출 파일에 배율을 적용한다 (make_submission.py --scale).
  참고로 평일 / 주말·공휴일 비율도 출력 (주말·공휴일 한낮에만 크게 과대면 봄철 출력제어로 보임 → 계절성일 수 있음).

실행 (2026_OIBC 폴더에서):
  python scoring/calibrate.py --round 3 cheun_transformer_v5 ensemble_lstm50_tf50_v4 [--tag h]
필요 파일: outputs/round{N}/preds/{모델}_dev_predictions.csv, {모델}_holdout_predictions.csv,
          outputs/round{N}/dev_opt/composition_{모델}_dev[_tag].csv (없으면 기존 A/B로 채점)
저장: outputs/round{N}/calibration/calibration.csv (모델별 한 줄, 실행할 때마다 갱신)
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import score as sc


def load(rdir, model, part):
    p = pd.read_csv(rdir / "preds" / f"{model}_{part}_predictions.csv")
    p["site_id"] = p.site_id.astype(str)
    p["ts"] = sc.to_kst(pd.to_datetime(p.ts, utc=True))
    p["pred_kwh"] = p.pred_kwh.clip(lower=0)
    a = sc.load_actuals(sc.DATA_DIR, p.ts.min(), p.ts.max()); a["site_id"] = a.site_id.astype(str)
    e = a.merge(p, on=["site_id", "ts"], how="left", validate="one_to_one")
    if e.pred_kwh.isna().any():
        raise ValueError(f"{model} {part}: 실적은 있는데 예측이 없는 행 {int(e.pred_kwh.isna().sum()):,}개")
    return e


def ratio(e):
    return float(e.actual_kwh.sum() / e.pred_kwh.sum())


def penalty(e, comp, smp, scale=1.0):
    g = (e.assign(pred_kwh=e.pred_kwh * scale).merge(comp, on="site_id")
           .groupby(["group", "ts"], as_index=False)[["actual_kwh", "pred_kwh"]].sum().merge(smp, on="ts"))
    return float(0.10 * (g.smp * g.k * (g.pred_kwh - g.actual_kwh).abs()).sum())


# 2025년 3~5월 공휴일(대체공휴일 포함). 주말·공휴일 비율 계산용
HOLIDAYS = pd.to_datetime(["2025-03-01", "2025-03-03", "2025-05-01", "2025-05-05", "2025-05-06"]).date


def run(model, rdir, sites, smp, tag):
    dev, hold = load(rdir, model, "dev"), load(rdir, model, "holdout")
    cp = rdir / "dev_opt" / f"composition_{model}_dev{('_' + tag) if tag else ''}.csv"
    if cp.is_file():
        comp = pd.read_csv(cp); comp["site_id"] = comp.site_id.astype(str); comp_name = cp.name
    else:
        comp = sc.default_composition(sites); comp_name = "기존 A/B"
    allp = pd.concat([dev, hold])
    off = (allp.ts.dt.dayofweek >= 5) | allp.ts.dt.date.isin(HOLIDAYS)
    r = {"model": model, "배정": comp_name,
         "비율_3월": ratio(dev[dev.ts.dt.month == 3]), "비율_4월": ratio(dev[dev.ts.dt.month == 4]), "비율_5월": ratio(hold),
         "비율_평일": ratio(allp[~off]), "비율_주말공휴일": ratio(allp[off]),
         "배율_3~4월": ratio(dev), "배율_3~5월": ratio(allp)}
    s = r["배율_3~4월"]
    before, after = penalty(hold, comp, smp), penalty(hold, comp, smp, s)
    r.update({"5월_벌금_전": before, "5월_벌금_후(3~4월 배율)": after, "5월_변화_%": 100 * (after / before - 1)})
    # 주별 점검 (3~5월, 그룹 합산 후)
    g = (allp.merge(comp, on="site_id").groupby(["group", "ts"], as_index=False)[["actual_kwh", "pred_kwh"]].sum()
           .merge(smp, on="ts"))
    g = g[g.ts < pd.Timestamp("2025-06-01 00:00", tz="Asia/Seoul")]          # 마지막 1시간(6/1 00시) 제외
    g["wk"] = g.ts.dt.isocalendar().week.values
    pw = lambda x, sc_: 0.10 * (x.smp * x.k * (x.pred_kwh * sc_ - x.actual_kwh).abs()).sum()
    for x in sorted({0.97, 0.975, 0.98, 0.99, round(s, 3)}):
        ch = g.groupby("wk")[["smp", "k", "pred_kwh", "actual_kwh"]].apply(lambda w: pw(w, x) / pw(w, 1.0) - 1)
        r[f"주별감소_x{x:g}"] = int((ch < 0).sum()); r["주_수"] = int(len(ch))
        r[f"3~5월_변화_%_x{x:g}"] = 100 * (pw(g, x) / pw(g, 1.0) - 1)
    weeks_ok = r[f"주별감소_x{round(s, 3):g}"] >= int(np.ceil(r["주_수"] * 2 / 3))
    r["5월_검증_통과"] = bool(after < before and weeks_ok)
    r.update({f"5월_벌금_x{x:g}": penalty(hold, comp, smp, x) for x in (0.95, 0.96, 0.97, 0.98, 0.99, 1.0)})
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--tag", default="h", help="채점에 쓸 배정 꼬리표 (기본 h, 없으면 composition_{모델}_dev.csv, 그것도 없으면 기존 A/B)")
    a = ap.parse_args()
    rdir = ROOT / "outputs" / f"round{a.round}"
    sites = pd.read_csv(sc.DATA_DIR / "sites.csv"); sites["site_id"] = sites.site_id.astype(str)
    smp = sc.load_smp(sc.DATA_DIR / "train" / "smp.csv")
    rows = []
    for m in a.models:
        tag = a.tag if (rdir / "dev_opt" / f"composition_{m}_dev_{a.tag}.csv").is_file() else ""
        rows.append(run(m, rdir, sites, smp, tag))
    out = rdir / "calibration" / "calibration.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        table = pd.concat([old[~old.model.isin(table.model)], table], ignore_index=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"{'모델':26s} {'3월':>6s} {'4월':>6s} {'5월':>6s} | {'평일':>6s} {'주말·공휴일':>8s} | 3~4월 배율 → 5월 벌금 변화 | 14주 중 벌금 준 주 | 검증")
    for r in rows:
        x = round(r["배율_3~4월"], 3)
        print(f"{r['model']:26s} {r['비율_3월']:6.3f} {r['비율_4월']:6.3f} {r['비율_5월']:6.3f} | {r['비율_평일']:6.3f} {r['비율_주말공휴일']:8.3f} | "
              f"{x:.3f} → {r['5월_변화_%']:+6.2f}%         | {r[f'주별감소_x{x:g}']:2d}/{r['주_수']}주            | {'통과' if r['5월_검증_통과'] else '실패'}")
    print("\n배율별 3~5월 벌금 변화 (14주 중 벌금 준 주):")
    for r in rows:
        print(f"  {r['model']:26s} " + " | ".join(f"x{x:g}: {r[f'3~5월_변화_%_x{x:g}']:+.2f}% ({r[f'주별감소_x{x:g}']}주)" for x in (0.97, 0.975, 0.98, 0.99)))
    print("\n읽는 법: 비율 = 실제÷예측 (1보다 작으면 예측이 많았던 것). 3~4월로 정한 배율이 5월 벌금도 줄이고, 14주 중 10주 이상에서 줄이면 '통과'.")
    print("         통과한 모델만 제출에 배율 적용: python submit/make_submission.py --round N 모델 --tag h --scale 배율")
    print(f"\n저장: {out.relative_to(ROOT)}")
