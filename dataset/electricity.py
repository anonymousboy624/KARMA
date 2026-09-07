"""
electricity_data_prep.py
-------------------------
Data preparation script for electricity load diagrams dataset for time series forecasting with deep learning.

This script:
1. Loads the raw electricity data
2. Handles missing values and preprocessing
3. Creates sliding window sequences for forecasting
4. Splits into train/validation/test sets
5. Saves processed data for model training

Usage:
    python electricity_data_prep.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os

# Configuration
DATA_PATH = "/workspaces/KARMA-/data/raw/electricityloaddiagrams/LD2011_2014.txt"
OUTPUT_DIR = "/workspaces/KARMA-/data/generated/electricity"
SEQUENCE_LENGTH = 48  # 12 hours (15-min intervals)
PREDICTION_HORIZON = 1  # Predict next 15 minutes
TEST_SIZE = 0.2
VAL_SIZE = 0.1
N_SAMPLES = (
    70000  # Rows to load from the raw file (skip leading zeros by starting at 2012)
)
N_FEATURES = 20  # Use only first 20 meters instead of 370
SKIP_LEADING_ZEROS = (
    True  # Drop the all-zero rows at the start of the dataset (all of 2011)
)


def load_data():
    """Load the electricity dataset."""
    print("Loading electricity data...")
    df = pd.read_csv(
        DATA_PATH, sep=";", decimal=",", nrows=N_SAMPLES, usecols=range(N_FEATURES + 1)
    )  # +1 for timestamp

    # Parse timestamp
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.set_index(df.columns[0], inplace=True)
    df.index.name = "timestamp"

    if SKIP_LEADING_ZEROS:
        # The raw file records 2011 as all-zeros for most meters; real data starts 2012-01-01.
        # Keeping those rows makes X_train entirely zero, breaking discretisation.
        # Drop every row where ALL features are zero.
        non_zero_mask = (df.values != 0).any(axis=1)
        first_nz = non_zero_mask.argmax()
        df = df.iloc[first_nz:].copy()
        print(
            f"Skipped {first_nz} leading all-zero rows; data now starts at {df.index[0]}"
        )

    print(f"Loaded data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


def preprocess_data(df):
    """Preprocess the data: handle missing values, outliers, etc."""
    print("Preprocessing data...")

    # Check for actual NaN values (not zeros)
    print(f"NaN values before: {df.isna().sum().sum()}")

    # Forward fill missing values
    df = df.ffill()

    # Backward fill any remaining NaNs at the start
    df = df.bfill()

    # Remove any remaining NaN rows
    df = df.dropna()

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
        seq = data[i - seq_length : i]  # (seq_length, D)
        target = data[i : i + pred_horizon]  # (pred_horizon, D)

        sequences.append(seq)
        targets.append(target)

    X = np.array(sequences)  # (N, seq_length, D)
    y = np.array(targets)  # (N, pred_horizon, D)
    return X, y


def normalize_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Normalize the data using training set statistics."""
    print("Normalizing data...")

    # Reshape for scaler (flatten samples and time, keep features)
    # X_train shape: (N, seq_length, D) -> reshape to (N*seq_length, D)
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])  # (N*seq_length, D)
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    # Fit scaler on training data features
    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    # Transform and reshape back to original shape
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)

    # For targets, normalize using the same scaler
    # y_train shape: (N, pred_horizon, D) -> reshape to (N*pred_horizon, D)
    y_train_flat = y_train.reshape(-1, y_train.shape[-1])  # (N*pred_horizon, D)
    y_val_flat = y_val.reshape(-1, y_val.shape[-1])
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])

    y_train_scaled = scaler.transform(y_train_flat).reshape(y_train.shape)
    y_val_scaled = scaler.transform(y_val_flat).reshape(y_val.shape)
    y_test_scaled = scaler.transform(y_test_flat).reshape(y_test.shape)

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        scaler,
    )


def save_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler):
    """Save processed data to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Saving processed data...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Save scaler
    import joblib

    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    # Save metadata
    metadata = {
        "sequence_length": SEQUENCE_LENGTH,
        "prediction_horizon": PREDICTION_HORIZON,
        "n_features": X_train.shape[-1],  # D dimension
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "data_shape": {
            "X_train": X_train.shape,
            "y_train": y_train.shape,
            "X_val": X_val.shape,
            "y_val": y_val.shape,
            "X_test": X_test.shape,
            "y_test": y_test.shape,
        },
        "format": "(N, seq_length, D)",  # Note the format
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Data saved to {OUTPUT_DIR}")
    print(f"Metadata: {metadata}")


def main():
    """Main data preparation pipeline."""
    print("Starting electricity data preparation for time series forecasting...")

    # Load data
    df = load_data()

    # Preprocess
    df_processed = preprocess_data(df)

    # Convert to numpy with float32
    data = df_processed.values.astype(np.float32)  # (T, D)

    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(data, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    print(f"Created {len(X)} sequences")

    # Split into train/val/test
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )

    # Second split: train vs val
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, shuffle=False
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize
    (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        scaler,
    ) = normalize_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # Save
    save_data(
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
        scaler,
    )

    print("Data preparation completed successfully!")


if __name__ == "__main__":
    main()
