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
python group/search.py          --round N 모델이름 --pick-by-holdout  # ② 3~4월 오차로 배정 후보 11개 탐색 (약 7분)
python group/check_holdout.py   --round N 모델이름 --tag h  # ③ 5월 예측에 배정 적용해 '이론상 최저와의 거리' 확인
python submit/make_submission.py --round N 모델이름 --tag h # ④ test 예측을 배정대로 합산 → 제출 파일 + 형식 검사
python submit/final_check.py outputs/roundN/submissions/모델이름_OPT_h   # ⑤ 제출 직전: 형식·출처·5월 확인 한 번에
```

②는 출발점 11개(기존 A/B + 무작위 10개)에서 각각 3~4월 페널티가 더 안 줄 때까지 내려간 배정 후보 11개를 만들고
`candidates_{모델}_dev.csv`에 모두 저장합니다. 후보 중 고르는 기준은 두 가지를 같이 저장합니다.

| 파일 | 고르는 기준 | 언제 쓰나 |
|---|---|---|
| `composition_{모델}_dev.csv` | 3~4월 페널티가 가장 작은 후보 | 1·2회차에 쓴 기존 방식 |
| `composition_{모델}_dev_h.csv` (`--pick-by-holdout`) | 5월 페널티가 '이론상 최저(한 그룹에 다 넣은 값)'에 가장 가까운 후보 | 3회차부터 제출 (`--tag h`) |

5월은 배정을 찾는 데는 쓰지 않고(탐색은 3~4월 오차만 사용), 이미 만들어진 후보 11개 중 하나를 고르는 데만 씁니다.
3~4월에서 가장 좋은 후보가 5월에서도 가장 좋은 경우는 드물었고(2회차 교훈: 3~4월을 더 깊게 맞출수록 리더보드는 나빠짐),
5월 거리는 1회차 리더보드 순위와 방향이 맞았습니다.

```
outputs/round{N}/
├─ preds/         받은 예측 ({모델}_dev/holdout/test_predictions.csv)
├─ dev_opt/       오차표, candidates_{모델}_dev.csv(후보 11개), composition_{모델}_dev[_h].csv/.json, model_scores_dev.csv
├─ holdout/       holdout_check.csv
└─ submissions/   {모델}_OPT/{모델}_OPT_composition.csv, {모델}_OPT_forecast.csv, info.json
```

- 배정은 3~4월(dev)에서만 찾습니다. 5월(holdout)은 후보 중 고르기·확인용, 평가 기간(test)은 정답이 없어 적용만 합니다.
- 제출 파일 이름에 모델명을 붙여 제출 내역에서 구분합니다 (2회차부터).

## 회차 기록

| 회차 | 날짜 | 모델 | 문서 |
| --- | --- | --- | --- |
| 1 | 11/7 | LSTM, Transformer(채은님), XGBoost | 5 |
| 2 | 11/8 | 1회차 모델 3개 + 탐색 강화 배정 (`search_deep.py`). 앙상블 3개는 준비만 하고 미제출 | 6 |
