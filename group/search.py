"""
그룹 배정 탐색: 발전소별 과거 검증 오차로 페널티가 가장 작은 G1/G2 배정을 찾는다.

원리
  시각 t의 그룹 오차 = 그 그룹 발전소 오차(예측-실제)의 합.
  한 그룹 안에서 과대·과소 예측이 섞이면 서로 상쇄돼 페널티가 줄어든다.
  페널티 = 0.10 * Σ_t SMP_t * k_t * ( |G1 오차_t| + |G2 오차_t| )
  하한: 두 그룹 오차가 매 시각 같은 부호면 |G1|+|G2| = |전체 오차| → 이보다 작아질 수 없다.

방법 (지역 탐색)
  기존 A/B 배정과 무작위 배정 여러 개에서 출발해
  ① 발전소 1개 옮기기, ② 두 그룹에서 1개씩 맞바꾸기 중
  페널티가 가장 많이 줄고 두 그룹 모두 20,000kW 이상인 것을 반복한다.

과적합 점검
  검증 기간의 날짜를 홀수일/짝수일로 나눠, 홀수일로 찾은 배정을 짝수일에서 채점한다.
  (찾을 때 안 본 날에서도 기존 배정보다 나은지 확인)

실행 (2026_OIBC 폴더에서):
  python group/search.py --round 2 모델이름            # outputs/round2/dev_opt/{모델이름}_dev_errors.csv 를 읽음
  python group/search.py --baseline                    # outputs/baseline/baseline_dev_errors.csv
  옵션: --restarts 20 (무작위 출발점 수), --seed 1
        --pick-by-holdout : 후보 배정(출발점별 결과)을 5월 예측으로 채점해, 5월 이론상 최저와 가장 가까운 것을 고름.
                            5월은 배정을 찾는 데는 쓰지 않고 후보 중 고르는 데만 씀. 결과는 composition_{모델}_dev_h.csv
저장 (오차표와 같은 폴더):
  candidates_{모델이름}_dev.csv      (후보 배정 전부: site_id, start00(기존 A/B 출발) ~ start10)
  composition_{모델이름}_dev.csv     (3~4월 페널티 최소 후보: site_id, group)
  composition_{모델이름}_dev.json    (점수 요약)
  composition_{모델이름}_dev_h.csv/.json  (--pick-by-holdout: 5월 최저와의 거리 최소 후보, 후보별 점수표 포함)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
import score as sc

MIN_CAP = 20000.0
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def load(errors_path):
    """오차표 → 발전소 x 시각 오차 행렬 E, 시각별 가중치 w(=SMP*k), 설비용량."""
    df = pd.read_csv(errors_path, usecols=["site_id", "ts", "actual_kwh", "pred_kwh"])
    df["ts"] = sc.to_kst(pd.to_datetime(df["ts"], utc=True))
    if df.duplicated(["site_id", "ts"]).any():
        raise ValueError("같은 (site_id, ts)가 두 번 이상 있어요")
    sites = pd.read_csv(sc.DATA_DIR / "sites.csv")
    smp = sc.load_smp(sc.DATA_DIR / "train" / "smp.csv")
    E = (df.assign(e=df.pred_kwh - df.actual_kwh)
           .pivot(index="site_id", columns="ts", values="e"))
    if E.isna().any().any():
        raise ValueError("빠진 (발전소, 시각)이 있어요. 모든 시각에 253개소가 있어야 해요")
    E = E.reindex(sites.site_id)
    if E.isna().any().any():
        raise ValueError("sites.csv의 발전소가 오차표에 다 있어야 해요")
    w = smp.set_index("ts").reindex(E.columns)
    if w.isna().any().any():
        raise ValueError("SMP가 없는 시각이 있어요")
    return df, sites, smp, E.to_numpy(float), (w.smp * w.k).to_numpy(float), \
        sites.capacity_kw.to_numpy(float), pd.DatetimeIndex(E.columns)


def penalty(a, tot, w):
    """a: G1 오차(시각별), tot: 전체 오차 → 페널티 (여러 후보를 한 번에 계산 가능)."""
    return 0.10 * ((np.abs(a) + np.abs(tot - a)) * w).sum(axis=-1)


def local_search(g1, E, w, cap, total_cap):
    """g1: G1이면 True. 옮기기·맞바꾸기로 더 이상 안 줄 때까지 개선."""
    g1 = g1.copy()
    tot = E.sum(0)
    a = E[g1].sum(0)
    c1 = cap[g1].sum()
    best = penalty(a, tot, w)
    while True:
        # ① 1개 옮기기
        sign = np.where(g1, -1.0, 1.0)                   # G1→G2면 빼고, G2→G1이면 더함
        cand = penalty(a[None, :] + sign[:, None] * E, tot, w)
        new_c1 = c1 + sign * cap
        ok = (new_c1 >= MIN_CAP) & (total_cap - new_c1 >= MIN_CAP)
        cand[~ok] = np.inf
        i = int(np.argmin(cand))
        if cand[i] < best - 1e-6:
            g1[i] = ~g1[i]; a = a + sign[i] * E[i]; c1 = new_c1[i]; best = cand[i]
            continue
        # ② 맞바꾸기 (G1의 i ↔ G2의 j)
        I, J = np.where(g1)[0], np.where(~g1)[0]
        found = None
        for i in I:
            cand = penalty((a - E[i])[None, :] + E[J], tot, w)
            new_c1 = c1 - cap[i] + cap[J]
            cand[(new_c1 < MIN_CAP) | (total_cap - new_c1 < MIN_CAP)] = np.inf
            k = int(np.argmin(cand))
            if cand[k] < best - 1e-6 and (found is None or cand[k] < found[0]):
                found = (cand[k], i, J[k], new_c1[k])
        if found is None:
            return g1, best
        best, i, j, c1 = found
        g1[i], g1[j] = False, True
        a = a - E[i] + E[j]


def random_start(rng, cap, total_cap):
    while True:
        g1 = rng.random(len(cap)) < 0.5
        if cap[g1].sum() >= MIN_CAP and total_cap - cap[g1].sum() >= MIN_CAP:
            return g1


def search(E, w, cap, start, restarts, seed, return_all=False):
    total_cap = cap.sum()
    rng = np.random.default_rng(seed)
    starts = [start] + [random_start(rng, cap, total_cap) for _ in range(restarts)]
    results = []                                   # (출발점 번호, 배정, 3~4월 페널티)
    best_g1, best_p = None, np.inf
    for n, s in enumerate(starts):
        g1, p = local_search(s, E, w, cap, total_cap)
        log(f"  출발점 {n:2d} ({'기존 A/B' if n == 0 else '무작위'}): 페널티 {p:,.0f}")
        results.append((n, g1, p))
        if p < best_p:
            best_g1, best_p = g1, p
    return (best_g1, best_p, results) if return_all else (best_g1, best_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, help="제출 회차 (outputs/round{N}/dev_opt/)")
    ap.add_argument("model", nargs="?", help="모델 이름 ({이름}_dev_errors.csv)")
    ap.add_argument("--baseline", action="store_true", help="outputs/baseline/baseline_dev_errors.csv 사용")
    ap.add_argument("--restarts", type=int, default=10, help="무작위 출발점 개수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pick-by-holdout", action="store_true",
                    help="후보 배정들을 5월로 채점해 이론상 최저와 가장 가까운 것을 고름 (결과 파일에 _h 꼬리표)")
    args = ap.parse_args()
    if args.baseline:
        path = ROOT / "outputs" / "baseline" / "baseline_dev_errors.csv"
    elif args.round and args.model:
        path = ROOT / "outputs" / f"round{args.round}" / "dev_opt" / f"{args.model}_dev_errors.csv"
    else:
        ap.error("--round N 모델이름  또는  --baseline 이 필요해요")
    name = path.stem.replace("_errors", "")

    df, sites, smp, E, w, cap, ts = load(path)
    tot = E.sum(0)
    ab = (sites.current_group == "A").to_numpy()
    lower = 0.10 * (np.abs(tot) * w).sum()
    log(f"{name}: 발전소 {E.shape[0]}개 x {E.shape[1]:,}시간")

    # 과적합 점검: 홀수일로 찾고 짝수일에서 채점
    odd = (ts.day % 2 == 1)
    log("과적합 점검: 홀수일로 탐색")
    g_odd, _ = search(E[:, odd], w[odd], cap, ab, args.restarts, args.seed)
    ev = ~odd
    check = {
        "탐색에 쓴 날": "홀수일", "채점한 날": "짝수일",
        "기존 A/B": float(penalty(E[ab][:, ev].sum(0), tot[ev], w[ev])),
        "찾은 배정": float(penalty(E[g_odd][:, ev].sum(0), tot[ev], w[ev])),
        "하한(한 그룹)": float(0.10 * (np.abs(tot[ev]) * w[ev]).sum()),
    }
    check["개선율_%"] = 100 * (1 - check["찾은 배정"] / check["기존 A/B"])
    log(f"  짝수일 페널티: 기존 A/B {check['기존 A/B']:,.0f} → 찾은 배정 {check['찾은 배정']:,.0f} "
        f"({check['개선율_%']:.1f}% 감소, 하한 {check['하한(한 그룹)']:,.0f})")

    # 전체 검증 기간으로 최종 탐색
    log("검증 기간 전체로 탐색")
    g1, p, cands = search(E, w, cap, ab, args.restarts, args.seed, return_all=True)

    # 후보 배정 전부 저장 (출발점별). 나중에 다른 기준으로 고를 수 있게
    cand_df = pd.DataFrame({"site_id": sites.site_id})
    for n, g, pen in cands:
        cand_df[f"start{n:02d}"] = np.where(g, "G1", "G2")
    cand_df.to_csv(path.parent / f"candidates_{name}.csv", index=False)

    def finish(g1, p, suffix, pick_note, extra=None):
        """배정 하나를 공통 점수 코드로 다시 채점·검사하고 composition_{name}{suffix}.csv/.json 으로 저장."""
        comp = pd.DataFrame({"site_id": sites.site_id, "group": np.where(g1, "G1", "G2")})
        sp = df.copy()
        sp["ts"] = sc.to_kst(sp["ts"])
        base = sc.score(sp, sc.default_composition(sites), smp)
        new = sc.score(sp, comp, smp)
        problems = sc.check_composition(comp, sites)
        if problems:
            raise ValueError(f"배정 규칙 위반: {problems}")
        if abs(new["penalty"] - p) > 1:
            raise ValueError(f"탐색 점수({p:,.0f})와 공통 점수 코드({new['penalty']:,.0f})가 달라요")
        out = path.parent
        fname = name + suffix
        comp.to_csv(out / f"composition_{fname}.csv", index=False)
        caps = comp.assign(cap=cap).groupby("group").cap.sum()
        moved = int((g1 != ab).sum())
        summary = {
            "오차표": str(path.relative_to(ROOT)),
            "선택기준": pick_note,
            "기간": f"{ts.min()} ~ {ts.max()}",
            "페널티_기존AB": base["penalty"], "페널티_찾은배정": new["penalty"],
            "페널티_하한(한 그룹)": lower,
            "개선율_%": 100 * (1 - new["penalty"] / base["penalty"]),
            "S_기존AB": base["S"], "S_찾은배정": new["S"],
            "그룹_용량_kW": caps.round(2).to_dict(),
            "그룹_발전소수": comp.group.value_counts().sort_index().to_dict(),
            "기존과_다른_발전소수": min(moved, len(g1) - moved),   # G1/G2 이름이 뒤바뀐 경우 고려
            "과적합_점검": check,
            "restarts": args.restarts, "seed": args.seed,
        }
        if extra:
            summary.update(extra)
        (out / f"composition_{fname}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        log(f"[{fname}] 페널티: 기존 A/B {base['penalty']:,.0f} → 찾은 배정 {new['penalty']:,.0f} "
            f"({summary['개선율_%']:.1f}% 감소, 하한 {lower:,.0f})")
        log(f"[{fname}] 그룹 용량 {summary['그룹_용량_kW']}, 저장: composition_{fname}.csv / .json")

    # ① 3~4월 페널티가 가장 작은 후보 (기존 방식) → composition_{모델}_dev.csv
    finish(g1, p, "", "3~4월 페널티 최소")

    # ② --pick-by-holdout: 후보 전부를 5월(holdout) 예측으로 채점해 5월 이론상 최저와 가장 가까운 것을 고름
    #    → composition_{모델}_dev_h.csv.  5월은 배정을 '찾는' 데는 쓰지 않고 후보 중 '고르는' 데만 씀.
    if args.pick_by_holdout:
        if args.baseline:
            raise SystemExit("--pick-by-holdout 은 --round N 모델이름 과 같이 써야 해요 (5월 예측 파일 필요)")
        hp = path.parent.parent / "preds" / f"{args.model}_holdout_predictions.csv"
        if not hp.is_file():
            raise FileNotFoundError(f"--pick-by-holdout 에 필요한 5월 예측이 없어요: {hp}")
        h = pd.read_csv(hp); h["site_id"] = h.site_id.astype(str); h["ts"] = sc.to_kst(pd.to_datetime(h.ts, utc=True))
        h["pred_kwh"] = h.pred_kwh.clip(lower=0)                       # check_holdout.py 와 같은 채점 방식
        if len(h) != 253 * 744 or h.duplicated(["site_id", "ts"]).any():
            raise ValueError(f"5월 예측은 253개소 × 744시간이어야 해요 (지금 {len(h):,}행)")
        ha = sc.load_actuals(sc.DATA_DIR, h.ts.min(), h.ts.max()); ha["site_id"] = ha.site_id.astype(str)
        he = ha.merge(h, on=["site_id", "ts"], how="left", validate="one_to_one")
        if he.pred_kwh.isna().any():
            raise ValueError("5월 실적은 있는데 예측이 없는 행이 있어요")
        HEdf = (he.assign(e=he.pred_kwh - he.actual_kwh).pivot(index="site_id", columns="ts", values="e")
                  .reindex(sites.site_id.astype(str)))
        HE = HEdf.to_numpy(float)
        hw_ = smp.set_index("ts").reindex(HEdf.columns); hw = (hw_.smp * hw_.k).to_numpy(float)
        if np.isnan(HE).any() or np.isnan(hw).any():
            raise ValueError("5월 오차 행렬 또는 SMP 에 빈 값이 있어요")
        htot = HE.sum(0); hlower = 0.10 * (np.abs(htot) * hw).sum()
        hab = penalty(HE[ab].sum(0), htot, hw)
        log(f"5월로 후보 {len(cands)}개 채점 (5월 이론상 최저 {hlower:,.0f}, 기존 A/B {hab:,.0f})")
        table = []
        for n, g, pen in cands:
            hpen = float(penalty(HE[g].sum(0), htot, hw))
            table.append({"출발점": n, "페널티_3~4월": float(pen), "페널티_5월": hpen, "5월_최저와의_거리": hpen - hlower})
            log(f"  출발점 {n:2d}: 3~4월 {pen:,.0f} | 5월 {hpen:,.0f} | 최저와의 거리 {hpen - hlower:,.0f}")
        best = min(table, key=lambda r: r["5월_최저와의_거리"])
        dev_best = min(table, key=lambda r: r["페널티_3~4월"])
        gh = next(g for n, g, _ in cands if n == best["출발점"])
        note = (f"5월 최저와의 거리 최소 (출발점 {best['출발점']}: 3~4월 {best['페널티_3~4월']:,.0f}, "
                f"5월 {best['페널티_5월']:,.0f}, 거리 {best['5월_최저와의_거리']:,.0f}; "
                f"3~4월 최소 후보는 출발점 {dev_best['출발점']}, 5월 거리 {dev_best['5월_최저와의_거리']:,.0f})")
        log(f"선택: {note}")
        finish(gh, best["페널티_3~4월"], "_h", note,
               extra={"5월_이론상_최저": hlower, "5월_기존AB": float(hab), "후보별_점수": table})


if __name__ == "__main__":
    main()
