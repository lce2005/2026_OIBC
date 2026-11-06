"""OIBC 2026 EXP-02: MLP extended-training model (2023-06 start) (Apple MPS when available).

Uses the same feature set and development dates as EXP-01 LightGBM.
Requires pandas, numpy, pyarrow and torch.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. Experiment settings
# ============================================================
EXP_ID = "exp02_mlp_train2306"
DATA_VERSION = "v0.1"
DATA_PATH = Path("data/train_v0.1.parquet")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
BASELINE_ERRORS = OUTPUT_DIR / "exp01_lgbm_train2306_dev_errors.csv"

TRAIN_START = pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul")
TRAIN_END = pd.Timestamp("2025-02-27 01:00", tz="Asia/Seoul")
DEV_START = pd.Timestamp("2025-03-01 01:00", tz="Asia/Seoul")
DEV_END = pd.Timestamp("2025-05-01 01:00", tz="Asia/Seoul")

BATCH_SIZE = 4096
EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 1e-3
SEED = 42

CAT_FEATURES = ["site_id", "wa_station", "wb_station"]
NUM_FEATURES = [
    "capacity_kw", "lat", "lon", "wa_dist", "wb_dist",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "elev", "azim", "cosz", "is_daylight",
    "wa_cloud_r10", "wa_cloud_r17",
    "wa_temp_r10", "wa_temp_r17",
    "wa_humidity_r10", "wa_humidity_r17",
    "wa_wind_speed_r10", "wa_wind_speed_r17",
    "wa_rain_r10", "wa_rain_r17",
    "wb_cloud_r10", "wb_cloud_r17",
    "wb_temp_r10", "wb_temp_r17",
    "wb_rel_hum_r10", "wb_rel_hum_r17",
    "wb_uv_idx_r10", "wb_uv_idx_r17",
    "wb_precip_prob_r10", "wb_precip_prob_r17",
    "wb_total_liq_r10", "wb_total_liq_r17",
    "wb_wind_speed_ms_r10", "wb_wind_speed_ms_r17",
    "fc_filled_a", "fc_filled_b", "fc_missing_a", "fc_missing_b",
]

LOAD_COLUMNS = list(dict.fromkeys(
    CAT_FEATURES + NUM_FEATURES + ["ts", "cf_target", "energy_kwh", "smp", "k"]
))


class SolarMLP(nn.Module):
    def __init__(self, n_numeric: int, vocab_sizes: list[int]):
        super().__init__()
        embedding_dims = [16, 8, 8]
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, dim, padding_idx=0)
            for vocab_size, dim in zip(vocab_sizes, embedding_dims)
        ])
        input_dim = n_numeric + sum(embedding_dims)
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, num, site, station_a, station_b):
        categorical = [site, station_a, station_b]
        parts = [num] + [
            embedding(values)
            for embedding, values in zip(self.embeddings, categorical)
        ]
        return self.network(torch.cat(parts, dim=1)).squeeze(1)


def preprocess_numeric(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    """Learn all imputation and normalization statistics on train ONLY."""
    tr = train_df[NUM_FEATURES].astype("float32")
    tr = tr.replace([np.inf, -np.inf], np.nan)
    medians = tr.median().fillna(0.0).astype("float32")
    x_train = tr.fillna(medians).to_numpy(dtype=np.float32, copy=True)
    del tr

    means = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    x_train -= means
    x_train /= stds
    np.clip(x_train, -8.0, 8.0, out=x_train)

    va = dev_df[NUM_FEATURES].astype("float32")
    va = va.replace([np.inf, -np.inf], np.nan)
    x_dev = va.fillna(medians).to_numpy(dtype=np.float32, copy=True)
    del va
    x_dev -= means
    x_dev /= stds
    np.clip(x_dev, -8.0, 8.0, out=x_dev)

    if not np.isfinite(x_train).all() or not np.isfinite(x_dev).all():
        raise ValueError("Numeric features still contain NaN/Inf")
    return x_train, x_dev, medians, means, stds


def preprocess_categories(train_df: pd.DataFrame, dev_df: pd.DataFrame):
    """Category zero is reserved for unknown/missing IDs."""
    train_codes, dev_codes, vocabularies = [], [], {}
    for col in CAT_FEATURES:
        categories = list(pd.Index(train_df[col].dropna().unique()))
        vocabularies[col] = [str(value) for value in categories]
        train_codes.append(
            (pd.Categorical(train_df[col], categories=categories).codes + 1)
            .astype(np.int64)
        )
        dev_codes.append(
            (pd.Categorical(dev_df[col], categories=categories).codes + 1)
            .astype(np.int64)
        )
    return train_codes, dev_codes, vocabularies


@torch.inference_mode()
def predict_cf(model, numerical, categorical, device, batch_size=8192):
    model.eval()
    result = np.empty(len(numerical), dtype=np.float32)
    for start in range(0, len(numerical), batch_size):
        end = min(start + batch_size, len(numerical))
        num = torch.from_numpy(numerical[start:end]).to(device)
        cats = [torch.from_numpy(arr[start:end]).to(device) for arr in categorical]
        result[start:end] = model(num, *cats).cpu().numpy()
    return result


def postprocess_kwh(cf_pred, capacity, daylight):
    return np.where(daylight == 0, 0.0, np.maximum(cf_pred, 0.0) * capacity)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{EXP_ID}] PyTorch {torch.__version__} / device={device}", flush=True)
    if device.type != "mps":
        print("WARNING: MPS unavailable; training will use CPU.", flush=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    started = time.perf_counter()
    print("[1/6] Loading training data...", flush=True)
    df = pd.read_parquet(DATA_PATH, columns=LOAD_COLUMNS)
    tr_mask = (df["ts"] >= TRAIN_START) & (df["ts"] < TRAIN_END)
    dv_mask = (df["ts"] >= DEV_START) & (df["ts"] < DEV_END)
    train_df = df.loc[tr_mask].copy()
    dev_df = df.loc[dv_mask].copy()
    del df, tr_mask, dv_mask
    gc.collect()

    if train_df.empty or dev_df.empty:
        raise ValueError("Train or dev split is empty; verify timestamps")
    if train_df["ts"].max() >= dev_df["ts"].min():
        raise ValueError("Train and dev dates overlap")
    for split_name, frame in [("train", train_df), ("dev", dev_df)]:
        if frame.duplicated(["site_id", "ts"]).any():
            raise ValueError(f"Duplicate (site_id,ts) in {split_name}")
    print(f"train={len(train_df):,} rows, dev={len(dev_df):,} rows")
    print(f"train sites={train_df.site_id.nunique()}, dev sites={dev_df.site_id.nunique()}")
    print("dev span:", dev_df.ts.min(), "to", dev_df.ts.max())

    print("[2/6] Fitting train-only preprocessing...", flush=True)
    x_train, x_dev, medians, means, stds = preprocess_numeric(train_df, dev_df)
    c_train, c_dev, vocabularies = preprocess_categories(train_df, dev_df)
    # Arrow-backed pandas arrays may be read-only; PyTorch needs a writable copy.
    y_train = train_df["cf_target"].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(y_train).all():
        raise ValueError("Non-finite training targets")
    actual = dev_df["energy_kwh"].to_numpy(dtype=np.float32)
    capacity = dev_df["capacity_kw"].to_numpy(dtype=np.float32)
    daylight = dev_df["is_daylight"].to_numpy()
    if not (np.isfinite(actual).all() and np.isfinite(capacity).all()):
        raise ValueError("Invalid actual/capacity values in dev")

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        *(torch.from_numpy(codes) for codes in c_train),
        torch.from_numpy(y_train),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
    )

    model = SolarMLP(
        n_numeric=len(NUM_FEATURES),
        vocab_sizes=[len(vocabularies[col]) + 1 for col in CAT_FEATURES],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.SmoothL1Loss(beta=0.05)

    print("[3/6] Training MLP...", flush=True)
    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        weighted_loss_sum = 0.0
        n_seen = 0
        for num, site, sta, stb, label in train_loader:
            num = num.to(device)
            site = site.to(device)
            sta = sta.to(device)
            stb = stb.to(device)
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(num, site, sta, stb)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()
            weighted_loss_sum += loss.item() * len(label)
            n_seen += len(label)

        val_cf = predict_cf(model, x_dev, c_dev, device)
        val_kwh = postprocess_kwh(val_cf, capacity, daylight)
        dev_mae = float(np.mean(np.abs(val_kwh - actual), dtype=np.float64))
        print(
            f"epoch {epoch:02d}/{EPOCHS}: "
            f"train_loss={weighted_loss_sum / n_seen:.6f}, "
            f"dev_MAE={dev_mae:.4f} kWh",
            flush=True,
        )
        if dev_mae < best_mae - 1e-5:
            best_mae = dev_mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    del best_state
    print(f"best_epoch={best_epoch}, best_dev_MAE={best_mae:.4f} kWh")

    print("[4/6] Evaluating best checkpoint...", flush=True)
    pred_cf = predict_cf(model, x_dev, c_dev, device)
    pred_kwh = postprocess_kwh(pred_cf, capacity, daylight)
    abs_error = np.abs(pred_kwh - actual)
    mae = float(abs_error.mean(dtype=np.float64))
    rmse = float(np.sqrt(np.mean((pred_kwh - actual) ** 2, dtype=np.float64)))
    day_mask = daylight == 1
    day_mae = float(abs_error[day_mask].mean(dtype=np.float64)) if day_mask.any() else None
    weights = dev_df["smp"].to_numpy(dtype=np.float64) * dev_df["k"].to_numpy(dtype=np.float64)
    proxy = float(0.10 * np.sum(weights * abs_error, dtype=np.float64))
    print(f"MAE={mae:.4f} kWh / RMSE={rmse:.4f} kWh / daylight MAE={day_mae}")
    print(f"Site-level weighted-error proxy={proxy:,.2f} (NOT official group penalty)")

    print("[5/6] Saving checkpoint and CSV files...", flush=True)
    ckpt_path = MODEL_DIR / f"{EXP_ID}.pt"
    torch.save({
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "num_features": NUM_FEATURES,
        "cat_features": CAT_FEATURES,
        "category_vocabularies": vocabularies,
        "numeric_medians": medians.to_dict(),
        "numeric_means": means.tolist(),
        "numeric_stds": stds.tolist(),
        "data_version": DATA_VERSION,
        "best_epoch": best_epoch,
        "train_start": str(TRAIN_START), "train_end_exclusive": str(TRAIN_END),
        "dev_start": str(DEV_START), "dev_end_exclusive": str(DEV_END),
    }, ckpt_path)

    errors = dev_df[["site_id", "ts", "energy_kwh"]].copy()
    errors = errors.rename(columns={"energy_kwh": "actual_kwh"})
    errors["pred_kwh"] = pred_kwh.astype(np.float32)
    errors["error_kwh"] = errors["pred_kwh"] - errors["actual_kwh"]
    errors = errors.sort_values(["site_id", "ts"])
    if errors.duplicated(["site_id", "ts"]).any():
        raise ValueError("Duplicate prediction keys")
    if not np.isfinite(errors["pred_kwh"]).all() or (errors["pred_kwh"] < 0).any():
        raise ValueError("NaN/Inf or negative predictions")

    pred_path = OUTPUT_DIR / f"{EXP_ID}_dev_predictions.csv"
    errors_path = OUTPUT_DIR / f"{EXP_ID}_dev_errors.csv"
    errors[["site_id", "ts", "pred_kwh"]].to_csv(pred_path, index=False)
    errors[["site_id", "ts", "actual_kwh", "pred_kwh", "error_kwh"]].to_csv(
        errors_path, index=False
    )

    metrics = {
        "experiment": EXP_ID, "data_version": DATA_VERSION, "device": str(device),
        "train_rows": len(train_df), "dev_rows": len(dev_df),
        "best_epoch": int(best_epoch),
        "mae_kwh": mae, "rmse_kwh": rmse, "day_mae_kwh": day_mae,
        "site_weighted_error_proxy": proxy,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    print("[6/6] Comparing with EXP-01 (if present)...", flush=True)
    if BASELINE_ERRORS.exists():
        base = pd.read_csv(BASELINE_ERRORS, usecols=["site_id", "ts", "pred_kwh"])
        base["ts"] = pd.to_datetime(base["ts"], utc=True).dt.tz_convert("Asia/Seoul")
        matched = errors.merge(
            base.rename(columns={"pred_kwh": "baseline_pred_kwh"}),
            on=["site_id", "ts"], how="inner", validate="one_to_one"
        )
        if len(matched) == len(errors) == len(base):
            # Convert pandas Series to NumPy before passing NumPy-only dtype=.
            baseline_error = (
                matched["baseline_pred_kwh"].to_numpy(dtype=np.float64)
                - matched["actual_kwh"].to_numpy(dtype=np.float64)
            )
            baseline_mae = float(np.mean(np.abs(baseline_error), dtype=np.float64))
            metrics["exp01_mae_kwh"] = baseline_mae
            print(f"EXP-01 MAE={baseline_mae:.4f}, EXP-02 MAE={mae:.4f} kWh")
        else:
            print("EXP-01 keys/rows differ; skipped apples-to-apples comparison.")
    else:
        print("EXP-01 CSV not found; skipped comparison.")

    metric_path = OUTPUT_DIR / f"{EXP_ID}_metrics.json"
    metric_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nFinished! Files:")
    for path in (ckpt_path, pred_path, errors_path, metric_path):
        print(" -", path)


if __name__ == "__main__":
    main()
