"""
그룹 배정 탐색 — 더 세게 (search.py와 같은 목표·규칙, 찾는 방법만 강화)

search.py와 다른 점
  1) 출발점을 많이: 기존 A/B 1개 + 무작위 N개 (기본 50개)
  2) 막히면 흔들기: 내려가다 멈추면 발전소 k개를 무작위로 반대 팀에 옮긴 뒤 다시 내려감 (기본 20번).
     더 좋아지면 거기서 계속, 아니면 원래 자리로 돌아와 다시 흔듦 → 작은 골짜기에서 빠져나오기 위한 방법
  3) 5월 확인은 하지 않음 → 결과 배정은 group/check_holdout.py 로 따로 확인

실행 (2026_OIBC 폴더에서):
  python group/search_deep.py --round 1 exp06_lstm_v3 --restarts 50 --kicks 20
  옵션 --tag deep : 결과 파일 이름 뒤에 붙임 (기본 "deep") → composition_{모델}_dev_deep.csv / .json
코랩: colab/search_deep_colab.py (같은 코드, 경로만 /content/oibc)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scoring"))
sys.path.insert(0, str(ROOT / "group"))
import score as sc
from search import load, penalty, local_search, random_start, MIN_CAP

t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def kick(g1, rng, cap, total_cap, k):
    """발전소 k개를 무작위로 반대 팀으로 옮김 (용량 조건을 지키는 것만)."""
    for _ in range(100):
        g = g1.copy()
        idx = rng.choice(len(g), size=k, replace=False)
        g[idx] = ~g[idx]
        c1 = cap[g].sum()
        if c1 >= MIN_CAP and total_cap - c1 >= MIN_CAP:
            return g
    return g1.copy()


def search_deep(E, w, cap, start, restarts, kicks, kick_size, seed, label=""):
    total_cap = cap.sum()
    rng = np.random.default_rng(seed)
    starts = [("기존 A/B", start)] + [(f"무작위 {i+1}", random_start(rng, cap, total_cap)) for i in range(restarts)]
    best_g1, best_p = None, np.inf
    for n, (name, s) in enumerate(starts):
        g1, p = local_search(s, E, w, cap, total_cap)
        # 흔들기
        for _ in range(kicks):
            g2, p2 = local_search(kick(g1, rng, cap, total_cap, kick_size), E, w, cap, total_cap)
            if p2 < p - 1e-6:
                g1, p = g2, p2
        if p < best_p:
            best_g1, best_p = g1, p
            log(f"  {label}출발점 {n:3d} ({name}): {p:,.0f}  ← 지금까지 최고")
        elif n % 10 == 0:
            log(f"  {label}출발점 {n:3d} ({name}): {p:,.0f}")
    return best_g1, best_p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("model")
    ap.add_argument("--restarts", type=int, default=50, help="무작위 출발점 수")
    ap.add_argument("--kicks", type=int, default=20, help="출발점마다 흔들기 횟수")
    ap.add_argument("--kick-size", type=int, default=8, help="한 번 흔들 때 옮기는 발전소 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="deep")
    ap.add_argument("--skip-oddeven", action="store_true", help="홀수/짝수일 점검 생략 (시간 절약)")
    a = ap.parse_args()

    rdir = ROOT / "outputs" / f"round{a.round}" / "dev_opt"
    path = rdir / f"{a.model}_dev_errors.csv"
    df, sites, smp, E, w, cap, ts = load(path)
    tot = E.sum(0)
    ab = (sites.current_group == "A").to_numpy()
    lower = 0.10 * (np.abs(tot) * w).sum()
    log(f"{a.model}: 발전소 {E.shape[0]}개 x {E.shape[1]:,}시간 | 출발점 {1 + a.restarts}개, 흔들기 {a.kicks}회 × {a.kick_size}개")

    check = None
    if not a.skip_oddeven:
        odd = (ts.day % 2 == 1); ev = ~odd
        log("홀수일로 탐색 → 짝수일 채점")
        g_odd, _ = search_deep(E[:, odd], w[odd], cap, ab, a.restarts, a.kicks, a.kick_size, a.seed, label="[홀수일] ")
        check = {"탐색에 쓴 날": "홀수일", "채점한 날": "짝수일",
                 "기존 A/B": float(penalty(E[ab][:, ev].sum(0), tot[ev], w[ev])),
                 "찾은 배정": float(penalty(E[g_odd][:, ev].sum(0), tot[ev], w[ev])),
                 "하한(한 그룹)": float(0.10 * (np.abs(tot[ev]) * w[ev]).sum())}
        check["개선율_%"] = 100 * (1 - check["찾은 배정"] / check["기존 A/B"])
        log(f"  짝수일: 기존 A/B {check['기존 A/B']:,.0f} → 찾은 배정 {check['찾은 배정']:,.0f} ({check['개선율_%']:.1f}% 감소)")

    log("검증 기간 전체로 탐색")
    g1, p = search_deep(E, w, cap, ab, a.restarts, a.kicks, a.kick_size, a.seed)
    comp = pd.DataFrame({"site_id": sites.site_id, "group": np.where(g1, "G1", "G2")})

    sp = df.copy(); sp["ts"] = sc.to_kst(sp["ts"])
    base = sc.score(sp, sc.default_composition(sites), smp)
    new = sc.score(sp, comp, smp)
    if sc.check_composition(comp, sites) or abs(new["penalty"] - p) > 1:
        raise ValueError("배정 규칙 위반 또는 점수 불일치")

    prev = rdir / f"composition_{a.model}_dev.json"
    prev_p = json.loads(prev.read_text(encoding="utf-8"))["페널티_찾은배정"] if prev.exists() else None

    name = f"composition_{a.model}_dev_{a.tag}"
    comp.to_csv(rdir / f"{name}.csv", index=False)
    summary = {"오차표": path.name, "방법": f"출발점 {1 + a.restarts}개 + 흔들기 {a.kicks}회×{a.kick_size}개",
               "기간": f"{ts.min()} ~ {ts.max()}",
               "페널티_기존AB": base["penalty"], "페널티_찾은배정": new["penalty"], "페널티_하한(한 그룹)": lower,
               "개선율_%": 100 * (1 - new["penalty"] / base["penalty"]),
               "기본탐색(search.py)_페널티": prev_p,
               "그룹_용량_kW": comp.assign(cap=cap).groupby("group").cap.sum().round(2).to_dict(),
               "그룹_발전소수": comp.group.value_counts().sort_index().to_dict(),
               "과적합_점검": check, "restarts": a.restarts, "kicks": a.kicks, "kick_size": a.kick_size, "seed": a.seed}
    (rdir / f"{name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    log(f"페널티: 기존 A/B {base['penalty']:,.0f} → 찾은 배정 {new['penalty']:,.0f} ({summary['개선율_%']:.1f}% 감소, 하한 {lower:,.0f})"
        + (f" | 기본 탐색 {prev_p:,.0f}보다 {prev_p - new['penalty']:,.0f}원 낮음" if prev_p else ""))
    log(f"그룹 용량 {summary['그룹_용량_kW']}, 저장: {name}.csv / .json")


if __name__ == "__main__":
    main()
