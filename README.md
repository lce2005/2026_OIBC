# 2026 OIBC — 김준영 (기준선·그룹 최적화·제출 파일)

브랜치 `junyoung/score`. 팀 노션 문서 1~6과 대응합니다.

## 폴더

| 폴더 | 내용 | 노션 문서 |
| --- | --- | --- |
| `eda/` | 발전량(정답) 점검, 공통 데이터 v0.4 발전량 처리 확인 | 1, 4 |
| `scoring/` | 공통 점수 계산(`score.py`), 예측 → 오차표(`make_errors.py`) | 2 |
| `baseline/` | 공식 baseline을 팀 확정 기간으로 돌려 기준 점수 산출 | 3 |
| `group/` | 그룹 배정 탐색(`search.py`), 탐색 강화(`search_deep.py`), 5월 확인(`check_holdout.py`) | 3, 5, 6 |
| `submit/` | 제출 파일 생성·형식 검사(`make_submission.py`), 제출 직전 최종 점검(`final_check.py`) | 5, 6 |
| `colab/` | 그룹 탐색 코랩용 (`search_colab.py`, `search_deep_colab.py`) | — |

데이터(`data/`)와 결과(`outputs/`)는 올리지 않습니다 (`.gitignore`).

## 제출 회차 절차 (문서 5·6)

모델 담당이 예측 3종(dev·holdout·test, `site_id, ts, pred_kwh`)을 주면 `outputs/round{N}/preds/`에 넣고 네 줄을 차례로 실행합니다.

```
python scoring/make_errors.py   --round N 모델이름   # ① 3~4월 오차표 + 기본 배정 벌금
python group/search.py          --round N 모델이름   # ② 3~4월 오차로 최적화 배정 탐색 (약 7분)
python group/check_holdout.py   --round N 모델이름   # ③ 5월 예측에 배정 적용해 기존 A/B와 비교
python submit/make_submission.py --round N 모델이름  # ④ test 예측을 배정대로 합산 → 제출 파일 + 형식 검사
python submit/final_check.py outputs/roundN/submissions/모델이름_OPT   # ⑤ 제출 직전: 형식·출처·5월 확인 한 번에
```

```
outputs/round{N}/
├─ preds/         받은 예측 ({모델}_dev/holdout/test_predictions.csv)
├─ dev_opt/       오차표, composition_{모델}_dev.csv/.json, model_scores_dev.csv
├─ holdout/       holdout_check.csv
└─ submissions/   {모델}_OPT/{모델}_OPT_composition.csv, {모델}_OPT_forecast.csv, info.json
```

- 배정은 3~4월(dev)에서만 찾습니다. 5월(holdout)은 확인용, 평가 기간(test)은 정답이 없어 적용만 합니다.
- 제출 파일 이름에 모델명을 붙여 제출 내역에서 구분합니다 (2회차부터).

## 회차 기록

| 회차 | 날짜 | 모델 | 문서 |
| --- | --- | --- | --- |
| 1 | 11/7 | LSTM, Transformer(채은님), XGBoost | 5 |
| 2 | 11/8 | 1회차 모델 3개 + 탐색 강화 배정 (`search_deep.py`). 앙상블 3개는 준비만 하고 미제출 | 6 |
