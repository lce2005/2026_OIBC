"""
제출 직전 최종 점검: 제출 폴더 하나를 넣으면 아래를 한 번에 확인한다.
  1. 형식 검사 (make_submission.py --check 와 같음)
  2. 출처: info.json에 적힌 test 예측·배정 파일이 실제로 있고, 제출 파일이 그 둘로 다시 만든 것과 같은지
  3. 배정 품질: 배정이 3~4월 탐색 결과(.json)와 일치하는지, 5월 확인(holdout_check.csv)을 거쳤는지, 5월에서 기존 A/B보다 나은지
  4. 모델 비교표(outputs/summary)에 이 모델이 있는지

실행 (2026_OIBC 폴더에서):
  python submit/final_check.py outputs/round2/submissions/cheun_transformer_OPT_deep
하나라도 [주의]가 있으면 제출 전에 사람이 확인한다.
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring")); sys.path.insert(0, str(ROOT / "submit"))
import score as sc


def rdir_of(path):
    """outputs/roundN/preds/x.csv → outputs/roundN"""
    return path.parent.parent
import make_submission as ms

notes = []
def ok(m): print(f"  [통과] {m}")
def warn(m): notes.append(m); print(f"  [주의] {m}")


def main(folder: Path):
    folder = folder if folder.is_absolute() else ROOT / folder
    print(f"최종 점검: {folder.relative_to(ROOT)}\n")

    print("1. 형식")
    ms.problems.clear()
    form_ok = ms.check(folder)
    if not form_ok:
        print("\n형식 검사 실패 → 제출 불가"); return False

    print("\n2. 출처")
    info_p = folder / "info.json"
    if not info_p.is_file():
        warn("info.json 없음 → 어떤 예측·배정으로 만든 파일인지 알 수 없음"); return finish()
    info = json.loads(info_p.read_text(encoding="utf-8"))
    preds_p, comp_p = ROOT / info["preds"], ROOT / info["composition"]
    for name, p in [("test 예측", preds_p), ("배정", comp_p)]:
        (ok if p.is_file() else warn)(f"{name}: {p.relative_to(ROOT)}" + ("" if p.is_file() else " (없음)"))
    if preds_p.is_file() and comp_p.is_file():
        sub_c = pd.read_csv(next(folder.glob("*composition.csv"))); sub_c["site_id"] = sub_c.site_id.astype(str)
        comp = pd.read_csv(comp_p); comp["site_id"] = comp.site_id.astype(str)
        m = sub_c.merge(comp, on="site_id", suffixes=("_sub", "_src"))
        (ok if (m.group_sub == m.group_src).all() else warn)("제출 composition = 배정 파일" + ("" if (m.group_sub == m.group_src).all() else " 아님!"))
        p = pd.read_csv(preds_p); p["site_id"] = p.site_id.astype(str); p["ts"] = sc.to_kst(pd.to_datetime(p.ts, utc=True))
        scale = float(info.get("scale", 1.0))
        f = (p.assign(pred_kwh=p.pred_kwh.clip(lower=0) * scale).merge(comp, on="site_id")
               .groupby(["group", "ts"]).pred_kwh.sum())
        sub_f = pd.read_csv(next(folder.glob("*forecast.csv"))); sub_f["ts"] = sc.to_kst(pd.to_datetime(sub_f.ts, utc=True))
        sub_f = sub_f.set_index(["group", "ts"]).forecast_kwh
        diff = float((f.reindex(sub_f.index) - sub_f).abs().max())
        (ok if diff < 1e-3 else warn)(f"제출 forecast = test 예측을 배정대로 합산한 값" + (f" × 배율 {scale}" if scale != 1.0 else "") + f" (최대 차이 {diff:.6f} kWh)")
        if scale != 1.0:
            cal = rdir_of(preds_p) / "calibration" / "calibration.csv"
            if cal.is_file():
                ct = pd.read_csv(cal); r = ct[ct.model == preds_p.name.replace("_test_predictions.csv", "")]
                if len(r) and r.iloc[0]["5월_검증_통과"] in (True, "True", 1):
                    ok(f"배율 {scale}: 5월 검증 통과 모델 (3~4월 배율 적용 시 5월 벌금 {r.iloc[0]['5월_변화_%']:+.2f}%, 권장 3~5월 배율 {r.iloc[0]['배율_3~5월']:.3f})")
                else:
                    warn(f"배율 {scale}: 이 모델은 5월 검증을 통과하지 못했거나 기록이 없음 (scoring/calibrate.py 확인)")
            else:
                warn(f"배율 {scale} 적용했지만 보정 기록(calibration.csv) 없음 → scoring/calibrate.py --round N 모델 실행 권장")
    print(f"  만든 시각: {info.get('made_at', '?')}")

    print("\n3. 배정 품질")
    model = comp_p.stem.replace("composition_", "").replace("_dev", "", 1) if comp_p.is_file() else None
    tag = ""
    if model and "_dev_" in comp_p.stem:
        base, tag = comp_p.stem.replace("composition_", "").split("_dev_", 1); model = base
    rdir = comp_p.parent.parent if comp_p.is_file() else None
    j = comp_p.with_suffix(".json")
    if j.is_file():
        s = json.loads(j.read_text(encoding="utf-8"))
        ok(f"3~4월 탐색 기록 있음: 기존 A/B {s['페널티_기존AB']:,.0f} → 최적화 {s['페널티_찾은배정']:,.0f} ({s['개선율_%']:.1f}% 감소)")
    else:
        warn("3~4월 탐색 기록(.json) 없음")
    hc = rdir / "holdout" / "holdout_check.csv" if rdir else None
    key = model + (f"_{tag}" if tag else "")
    if hc and hc.is_file() and key in set(pd.read_csv(hc).model):
        r = pd.read_csv(hc).set_index("model").loc[key]
        (ok if r.holdout_penalty_opt < r.holdout_penalty_AB else warn)(
            f"5월 확인: 기존 A/B {r.holdout_penalty_AB:,.0f} → 최적화 {r.holdout_penalty_opt:,.0f} ({r['reduction_%']:.1f}%)"
            + ("" if r.holdout_penalty_opt < r.holdout_penalty_AB else " → 5월에서 기존 A/B보다 나쁨!"))
    else:
        warn(f"5월 확인 기록 없음 (group/check_holdout.py --round N {model}" + (f" --tag {tag}" if tag else "") + " 실행 필요)")

    print("\n4. 비교표")
    summ = sorted((ROOT / "outputs" / "summary").glob("model_comparison_*.csv"))
    if summ:
        t = pd.read_csv(summ[-1], encoding="utf-8-sig")
        print(f"  최신 비교표: {summ[-1].name} ({len(t)}개 모델). 이 모델이 표에 있는지, 5월 기준 순위가 제출할 만한지 확인")
    else:
        warn("모델 비교표 없음")
    return finish()


def finish():
    print()
    if notes:
        print(f"[주의] {len(notes)}건 → 제출 전에 확인:"); [print("  - " + n) for n in notes]
    else:
        print("최종 점검 통과: 제출해도 됨")
    return not notes


if __name__ == "__main__":
    if len(sys.argv) < 2: print(__doc__); sys.exit(1)
    sys.exit(0 if main(Path(sys.argv[1])) else 1)
