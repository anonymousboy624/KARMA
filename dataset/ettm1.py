"""
ettm1.py
--------
Data preparation script for the ETT-m1 (Electricity Transformer Temperature,
15-minute sampling) dataset for time series forecasting with deep learning.

Usage:
    python dataset/ettm1.py
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = "/workspaces/KARMA-/data/generated/ettm1"
CSV_PATH = "/workspaces/KARMA-/data/raw/ETTm1.csv"

DATASET_NAME = "ettm1"
DISPLAY_NAME = "ETT-m1 (Electricity Transformer Temperature, 15-min)"

D = 7
FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

SEQUENCE_LENGTH = 48   # 12 hours of 15-min data
PREDICTION_HORIZON = 1

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10


def load_data() -> np.ndarray:
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["date"])
    df = df[FEATURE_NAMES]
    return df.values.astype(np.float32)


def create_sequences(data: np.ndarray, seq_length: int, pred_horizon: int):
    T, D = data.shape
    sequences, targets = [], []
    for i in range(seq_length, T - pred_horizon + 1):
        sequences.append(data[i - seq_length : i])
        targets.append(data[i : i + pred_horizon])
    return np.array(sequences), np.array(targets)


def main():
    print(f"=== {DISPLAY_NAME} data preparation ===")

    data = load_data()
    T = len(data)
    print(f"Loaded shape: {data.shape}")

    n_train = int(T * TRAIN_FRAC)
    n_val = int(T * VAL_FRAC)
    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]
    print(f"Split  train={len(train_data):,}  val={len(val_data):,}  test={len(test_data):,}")

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_sc = scaler.transform(train_data).astype(np.float32)
    val_sc = scaler.transform(val_data).astype(np.float32)
    test_sc = scaler.transform(test_data).astype(np.float32)

    X_train, y_train = create_sequences(train_sc, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_val, y_val = create_sequences(val_sc, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_test, y_test = create_sequences(test_sc, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    print(f"Sequences  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    print("Done.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
