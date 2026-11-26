# 공통 점수 계산 (scoring)

담당: 김준영 (기준선·집합자원 최적화)

대회 채점 방식을 그대로 따라 하는 코드입니다.
주최 측이 채점 코드를 배포하지 않아서, 제출하지 않고도 팀 안에서 점수를 미리 재보려고 만들었습니다.

## 사용법

레포 최상위 폴더에서 실행합니다.

```
python scoring/score.py 예측파일.csv                 # 기본 배정(A→G1, B→G2)으로 채점
python scoring/score.py 예측파일.csv 배정파일.csv     # 지정한 배정으로 채점
python scoring/score.py --test                       # 작은 예제로 계산 검산
```

| 입력 | 필요한 열 |
| --- | --- |
| 예측 파일 (csv 또는 parquet) | `site_id, ts, pred_kwh` |
| 배정 파일 (선택) | `site_id, group` (G1/G2) |

- 예측 파일에 들어 있는 기간을 채점합니다.
- 그 기간에 실적이 있는 (발전소, 시각)은 전부 예측이 있어야 합니다.
- 중복, 음수, 빈 값, 무한대가 있으면 오류로 알려줍니다.
- 시각은 타임존이 있는 형식 (예: `2025-03-01 01:00:00+09:00`)

**출력:** 정산금, 벌금, 가중치 없는 절대오차 합(동점 처리용)

## 예측 파일 → 오차표 (`make_errors.py`)

모델 담당이 준 예측 파일에 공통 정답을 붙여 팀 전달 형식 오차표를 만들고, 기본 배정 벌금을 계산합니다.
모든 모델의 정답이 한곳(`score.py`)에서 나오므로 비교가 어긋나지 않습니다.

```
python scoring/make_errors.py --round 2 모델이름 [모델이름 ...]
```

| 입력 | `outputs/round2/preds/{모델이름}_dev_predictions.csv` (`site_id, ts, pred_kwh`, 검증 기간) |
| --- | --- |
| 출력 | `outputs/round2/dev_opt/{모델이름}_dev_errors.csv` (`site_id, ts, actual_kwh, pred_kwh, error_kwh`) |
| 출력 | `outputs/round2/dev_opt/model_scores_dev.csv` (모델별 MAE·RMSE·기본 배정 벌금·정산금 표, 실행마다 갱신) |

## 계산식

```
정산금 S = Σ SMP × 실적 − 0.10 × Σ SMP × k × |예측 − 실적|
```

- 그룹별·시각별로 발전소 예측과 실적을 먼저 합산한 뒤 계산 → 같은 그룹 안에서는 오차가 상쇄됨
- k: ts의 시가 11~15이면 3, 아니면 1 (train/smp.csv에는 k가 없어 규칙으로 생성)
- 오차 정의: 예측 − 실제 (팀 공통 규칙)
- 실적의 음수는 0으로 처리

## 다른 코드에서 쓸 때

그룹 최적화 등 다른 코드에서는 함수를 가져다 씁니다.

| 함수 | 설명 |
| --- | --- |
| `score(site_preds, composition, smp)` | 실제값·예측값이 함께 있는 표로 채점 |
| `score_preds(preds, composition)` | 예측값만 있는 표로 채점 (실제값 자동 연결) |
| `breakdown(...)` | 벌금이 어느 달·어느 가중치 시간대에서 나왔는지 비율 |
| `check_composition(composition, sites)` | 배정 검사: 253개 중복·누락, G1/G2 이름, 각 그룹 20,000kW 이상 |

## 검산

- `--test`: 13시(k=3), SMP 100원, 발전소 2개 오차 +100 / −100kWh
  - 같은 그룹 벌금 0원, 다른 그룹 벌금 6,000원 → 손계산과 일치
- 같은 예측을 두 방식(`score`, `score_preds`)으로 채점해 결과 일치 확인

## 데이터 경로

아래 위치에 대회 데이터가 있다고 가정합니다. 데이터는 레포에 올리지 않습니다(`.gitignore`).

```
data/OIBC_2026_DATA/handover 2/01_participant_dataset/
```

## 필요 라이브러리

numpy, pandas, pyarrow