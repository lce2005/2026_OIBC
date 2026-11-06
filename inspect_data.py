import pandas as pd

train = pd.read_parquet("data/train_v0.1.parquet")
test = pd.read_parquet("data/test_v0.1.parquet")

print("===== TRAIN =====")
print("크기:", train.shape)
print("컬럼:", train.columns.tolist())
print(train.head())

print("\n===== TEST =====")
print("크기:", test.shape)
print("컬럼:", test.columns.tolist())
print(test.head())