import pandas as pd

df = pd.read_parquet(
    "data/train_v0.1.parquet",
    columns=[
        "site_id",
        "ts",
        "split",
        "cf_target",
        "energy_kwh",
        "is_suspect_zero"
    ]
)

print("===== 데이터 구간별 정보 =====")

summary = df.groupby("split", dropna=False).agg(
    rows=("site_id", "size"),
    sites=("site_id", "nunique"),
    start=("ts", "min"),
    end=("ts", "max")
)

print(summary.to_string())

print("\n===== 정답 결측 =====")
print(df["cf_target"].isna().sum())

print("\n===== 의심 발전량 =====")
print(df["is_suspect_zero"].value_counts(dropna=False))

print("\n===== 데이터 타입 =====")
print(df.dtypes)