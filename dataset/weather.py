"""
weather.py
----------
Data preparation script for the Weather dataset (21 meteorological indicators)
for time series forecasting with deep learning.

This script:
1. Reads weather.csv from data/raw/, drops the date column → numpy (T, 21) float32
2. Splits chronologically 70/10/20 (train/val/test)
3. Fits StandardScaler on train only
4. Creates sliding window sequences (SEQUENCE_LENGTH=24, PREDICTION_HORIZON=1)
5. Saves X_train/y_train/X_val/y_val/X_test/y_test + scaler.pkl + metadata.json

Usage:
    python dataset/weather.py
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configuration
OUTPUT_DIR = "/workspaces/KARMA-/data/generated/weather"
WEATHER_CSV_PATH = "/workspaces/KARMA-/data/raw/weather.csv"

SEQUENCE_LENGTH = 24       # input window (1 day at hourly resolution)
PREDICTION_HORIZON = 1     # predict next step
N_ROWS = 50000             # limit for speed
N_FEATURES = 21            # 21 meteorological indicators (all columns after date)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
# TEST_FRAC = 0.20  (remainder)


def load_data():
    """Load the weather dataset, drop the date column, limit to N_ROWS."""
    print("Loading weather data...")
    df = pd.read_csv(WEATHER_CSV_PATH, nrows=N_ROWS)

    # Drop the date/timestamp column (first column)
    date_col = df.columns[0]
    df = df.drop(columns=[date_col])

    # Keep only the first N_FEATURES numeric columns
    df = df.iloc[:, :N_FEATURES]

    print(f"Loaded data shape: {df.shape}")
    return df


def preprocess_data(df):
    """Handle missing values."""
    print("Preprocessing data...")
    print(f"NaN values before: {df.isna().sum().sum()}")
    df = df.ffill().bfill().dropna()
    print(f"After preprocessing: {df.shape}")
    return df


def create_sequences(data, seq_length, pred_horizon):
    """
    Create sliding window sequences for forecasting.

    Args:
        data: numpy array of shape (T, D)
        seq_length: length of input sequence
        pred_horizon: how many steps ahead to predict

    Returns:
        X: input sequences (N, seq_length, D)
        y: target values (N, pred_horizon, D)
    """
    T, D = data.shape
    sequences = []
    targets = []

    for i in range(seq_length, T - pred_horizon + 1):
        seq = data[i - seq_length : i]          # (seq_length, D)
        target = data[i : i + pred_horizon]      # (pred_horizon, D)
        sequences.append(seq)
        targets.append(target)

    X = np.array(sequences)   # (N, seq_length, D)
    y = np.array(targets)     # (N, pred_horizon, D)
    return X, y


def split_data(data):
    """Split raw time series chronologically into train / val / test."""
    T = len(data)
    n_train = int(T * TRAIN_FRAC)
    n_val = int(T * VAL_FRAC)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def scale_and_create_sequences(train_raw, val_raw, test_raw):
    """Fit StandardScaler on train, transform all splits, create windows."""
    print("Scaling data (fit on train only)...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw).astype(np.float32)
    val_scaled = scaler.transform(val_raw).astype(np.float32)
    test_scaled = scaler.transform(test_raw).astype(np.float32)

    print("Creating sliding-window sequences...")
    X_train, y_train = create_sequences(train_scaled, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_val, y_val = create_sequences(val_scaled, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    X_test, y_test = create_sequences(test_scaled, SEQUENCE_LENGTH, PREDICTION_HORIZON)

    print(
        f"Sequences — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def save_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    """Save processed arrays, scaler, and metadata to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Saving processed data...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    metadata = {
        "dataset": "weather",
        "display": "Weather (21 meteorological indicators)",
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "n_features": X_train.shape[-1],
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

    print(f"Data saved to {OUTPUT_DIR}")
    print(f"Metadata: {metadata}")


def main():
    """Main data preparation pipeline."""
    print("Starting Weather data preparation for time series forecasting...")

    df = load_data()
    df = preprocess_data(df)

    data = df.values.astype(np.float32)   # (T, 21)

    train_raw, val_raw, test_raw = split_data(data)

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = scale_and_create_sequences(
        train_raw, val_raw, test_raw
    )

    save_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler)

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
