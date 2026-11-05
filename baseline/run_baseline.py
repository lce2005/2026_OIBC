"""
공식 기준선(baseline.py)을 팀 확정 기간으로 다시 학습하고, 검증 기간 점수를 낸다.

팀 확정 기간 (11/4 카톡 확정, 경계는 주영님 데이터 v0.4의 split 열과 같음. KST, ts 기준)
  학습      2023-06-01 01:00 ~ 2025-03-01 00:00
  검증      2025-03-01 01:00 ~ 2025-05-01 00:00
  최종 검증 2025-05-01 01:00 ~ 2025-06-01 00:00

공식 baseline.py의 특징·모델 설정·라운드 10/17 평균을 그대로 쓰고, 학습 기간만 바꾼다.
실행 (2026_OIBC 폴더에서):
  python baseline/run_baseline.py              # 검증 기간 (기본)
  python baseline/run_baseline.py --period holdout   # 최종 검증 기간 (마지막에만)
  라운드별 예측은 outputs/baseline_{period}_round{10,17}.parquet에 저장되고, 있으면 재사용한다
  (학습 기간이나 모델을 바꾸면 이 파일들을 지우고 다시 실행).
저장:
  outputs/baseline_{period}_errors.csv  (site_id, ts, actual_kwh, pred_kwh, error_kwh)
  outputs/baseline_{period}_score.json
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import score as sc                      # 공통 점수 계산 코드
sys.path.insert(0, str(sc.DATA_DIR))
import baseline as official             # 대회 제공 baseline.py (수정 안 함)

KST = "Asia/Seoul"
T = lambda s: pd.Timestamp(s, tz=KST)
SPLIT = {
    "train":   (T("2023-06-01 01:00"), T("2025-03-01 00:00")),
    "dev":     (T("2025-03-01 01:00"), T("2025-05-01 00:00")),
    "holdout": (T("2025-05-01 01:00"), T("2025-06-01 00:00")),
}
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--one-round", action="store_true",
                    help="라운드 하나만 새로 학습하고 멈춤 (실행 시간 제한이 있을 때)")
    args = ap.parse_args()
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    D = sc.DATA_DIR
    (tr_s, tr_e), (ev_s, ev_e) = SPLIT["train"], SPLIT[args.period]

    sites = pd.read_csv(D / "sites.csv")[
        ["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]]
    gen = pd.read_parquet(D / "train" / "generation.parquet")
    gen["ts"] = sc.to_kst(gen["ts"])
    train = gen[(gen.ts >= tr_s) & (gen.ts <= tr_e)][["site_id", "ts", "energy"]]
    actual = sc.load_actuals(D, ev_s, ev_e)
    del gen
    log(f"학습 {len(train):,}행 ({train.ts.min()} ~ {train.ts.max()})")
    log(f"{args.period} {len(actual):,}행, {actual.ts.nunique():,}시간, {actual.site_id.nunique()}개소")

    def fc(name, cols, station):
        d = pd.read_parquet(D / "train" / f"forecast_{name}.parquet", columns=[station, "round", "ts"] + cols)
        d["ts"] = sc.to_kst(d["ts"])
        return d[(d.ts >= tr_s) & (d.ts <= ev_e)]
    fa = fc("wa", official.WA_COLS, "wa_station")
    fb = fc("wb", official.WB_COLS, "wb_station")

    index = pd.DatetimeIndex(sorted(actual.ts.unique()))
    preds = []
    for r in official.ROUNDS:
        cache = out / f"baseline_{args.period}_round{r}.parquet"   # 라운드별 예측 저장본
        if cache.exists():
            preds.append(pd.read_parquet(cache)); log(f"round {r}: 저장된 예측 사용"); continue
        d = official.features(train, sites, fa, fb, None, r)
        d = d[d.elev > 0]
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
        X = d[official.FEATS].astype("float32")
        del d
        log(f"round {r}: 학습 {X.shape}")
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06, num_leaves=127,
                              min_child_samples=200, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1, n_jobs=-1).fit(X, y)
        del X, y
        e = official.features(None, sites, fa, fb, index, r)
        p = np.where(e.elev.to_numpy() > 0, m.predict(e[official.FEATS].astype("float32")), 0.0)
        e["pred_kwh"] = np.clip(p, 0, None) * e.cap.to_numpy()
        e[["site_id", "ts", "pred_kwh"]].to_parquet(cache, index=False)
        preds.append(e[["site_id", "ts", "pred_kwh"]])
        log(f"round {r}: 예측 완료 → {cache.name}")
        del e, m
        if args.one_round and len(preds) < len(official.ROUNDS):
            log("--one-round: 여기서 멈춤. 다시 실행하면 이어서 진행"); return

    pred = pd.concat(preds).groupby(["site_id", "ts"], as_index=False).pred_kwh.mean()
    pred["ts"] = sc.to_kst(pred["ts"])
    comp = sc.default_composition(sites)
    result = sc.score_preds(pred, comp)          # 기존 A/B 배정으로 채점

    err = actual.merge(pred, on=["site_id", "ts"], how="left", validate="one_to_one")
    err["error_kwh"] = err.pred_kwh - err.actual_kwh
    err.to_csv(out / f"baseline_{args.period}_errors.csv", index=False)
    result.update(period_name=args.period, train=f"{tr_s} ~ {tr_e}",
                  composition="sites.csv current_group (A→G1, B→G2)",
                  mae_kwh=float(err.error_kwh.abs().mean()))
    (out / f"baseline_{args.period}_score.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("결과: " + json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
