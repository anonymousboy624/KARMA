"""
ettm2.py
--------
Data preparation script for the ETTm2 (Electricity Transformer Temperature,
15-minute sampling) dataset for time series forecasting with deep learning.

This script:
1. Reads ETTm2.csv from data/raw/, drops the date column → numpy (T, 7) float32
2. Splits chronologically 70/10/20 (train/val/test)
3. Fits a StandardScaler on train only
4. Creates sliding windows (SEQUENCE_LENGTH=48, PREDICTION_HORIZON=1)
5. Saves X_train/y_train/X_val/y_val/X_test/y_test + scaler + metadata.json

Usage:
    python dataset/ettm2.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET_NAME = "ettm2"
DISPLAY_NAME = "ETT-m2 (Electricity Transformer Temperature, 15-min)"

OUTPUT_DIR = "/workspaces/KARMA-/data/generated/ettm2"
CSV_PATH = "/workspaces/KARMA-/data/raw/ETTm2.csv"

# Features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT  (7 columns after dropping date)
D = 7
FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

SEQUENCE_LENGTH = 48  # 12 hours of 15-min data
PREDICTION_HORIZON = 12

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
# TEST_FRAC  = 0.20  (remainder)


def load_data() -> np.ndarray:
    """Load ETTm2 CSV, drop date column → (T, 7) float32."""
    print("Loading ETTm2 data...")
    df = pd.read_csv(CSV_PATH)
    # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    df = df.drop(columns=["date"])
    print(f"Loaded shape (after dropping date): {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    data = df.values.astype(np.float32)  # (T, 7)
    return data


def create_sequences(data: np.ndarray, seq_length: int, pred_horizon: int):
    """
    Create sliding window sequences.

    Parameters
    ----------
    data        : (T, D) numpy array
    seq_length  : length of input window
    pred_horizon: number of steps ahead to predict

    Returns
    -------
    X : (N, seq_length, D)
    y : (N, pred_horizon, D)
    """
    T, D = data.shape
    sequences, targets = [], []
    for i in range(seq_length, T - pred_horizon + 1):
        sequences.append(data[i - seq_length : i])  # (seq_length, D)
        targets.append(data[i : i + pred_horizon])  # (pred_horizon, D)
    X = np.array(sequences)  # (N, seq_length, D)
    y = np.array(targets)  # (N, pred_horizon, D)
    return X, y


def normalize_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Fit StandardScaler on train, apply to all splits."""
    print("Normalising data...")

    scaler = StandardScaler()
    # Fit on flattened train inputs
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    X_train_sc = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
        X_train.shape
    )
    X_val_sc = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_sc = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
        X_test.shape
    )

    y_train_sc = scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(
        y_train.shape
    )
    y_val_sc = scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)
    y_test_sc = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(
        y_test.shape
    )

    return X_train_sc, y_train_sc, X_val_sc, y_val_sc, X_test_sc, y_test_sc, scaler


def save_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    """Save arrays, scaler and metadata to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving processed data to {OUTPUT_DIR} ...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    import joblib

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    metadata = {
        "dataset": DATASET_NAME,
        "display_name": DISPLAY_NAME,
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "n_features": D,
        "feature_names": FEATURE_NAMES,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "data_shape": {
            "X_train": list(X_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(X_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(X_test.shape),
            "y_test": list(y_test.shape),
        },
        "format": "(N, seq_length, D)",
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved.  Metadata:")
    print(json.dumps(metadata, indent=2))


def main():
    print(f"=== ETTm2 data preparation ===")

    data = load_data()  # (T, 7)
    T = len(data)
    print(f"Total time steps: {T:,}")

    # ── Chronological split ────────────────────────────────────────────────────
    n_train = int(T * TRAIN_FRAC)
    n_val = int(T * VAL_FRAC)
    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]
    print(
        f"Split sizes  train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,}"
    )

    # ── Create sliding windows ─────────────────────────────────────────────────
    print("Creating sequences...")
    X_train, y_train = create_sequences(train_data, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_val, y_val = create_sequences(val_data, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_test, y_test = create_sequences(test_data, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    print(
        f"Sequences  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
    )

    # ── Normalise ──────────────────────────────────────────────────────────────
    (
        X_train_sc,
        y_train_sc,
        X_val_sc,
        y_val_sc,
        X_test_sc,
        y_test_sc,
        scaler,
    ) = normalize_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_data(X_train_sc, y_train_sc, X_val_sc, y_val_sc, X_test_sc, y_test_sc, scaler)
    print("ETTm2 data preparation completed successfully!")


if __name__ == "__main__":
    main()
