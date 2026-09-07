"""
web_traffic_data_prep.py
------------------------
Data preparation script for Kaggle Wikipedia Web Traffic dataset for time series forecasting with deep learning.

This script:
1. Loads the raw web traffic data from TSF format
2. Handles multivariate time series (multiple Wikipedia pages)
3. Creates sliding window sequences for forecasting
4. Splits into train/validation/test sets
5. Saves processed data for model training

Usage:
    python dataset/web_traffic.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os

# Configuration
DATA_PATH = "/workspaces/KARMA-/data/raw/web_traffic/kaggle_web_traffic_dataset_without_missing_values.tsf"
OUTPUT_DIR = "/workspaces/KARMA-/data/generated/web_traffic"
SEQUENCE_LENGTH = 14  # 2 weeks of daily data for input
PREDICTION_HORIZON = 7  # Predict next 1 week (7 days)
TEST_SIZE = 0.2
VAL_SIZE = 0.1
N_SERIES = 100  # Use only first 100 time series instead of all 145,063
MAX_LENGTH = 730  # Use approximately 730 days (2 years) of data per series


def load_tsf_file(path, num_series=None, max_length=None):
    """
    Load TSF file format for web traffic data.

    Returns:
        data: list of time series (each as numpy array)
        names: list of series names
    """
    print("Loading web traffic data from TSF file...")

    data = []
    names = []
    series_count = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip header lines and empty lines
            if not line or line.startswith("@") or line.startswith("#"):
                continue

            # Parse data lines (format: SeriesName:timestamp:values)
            if ":" in line:
                parts = line.split(":")
                if len(parts) >= 3:
                    series_name = parts[0]
                    # timestamp = parts[1]
                    values_str = parts[2]

                    # Parse comma-separated values
                    try:
                        values = np.array(
                            [float(v) for v in values_str.split(",")], dtype=np.float32
                        )

                        # Limit length if specified
                        if max_length is not None and len(values) > max_length:
                            values = values[-max_length:]

                        # Only keep series with sufficient data
                        if len(values) >= SEQUENCE_LENGTH + PREDICTION_HORIZON:
                            data.append(values)
                            names.append(series_name)
                            series_count += 1

                            # Stop if we have enough series
                            if num_series is not None and series_count >= num_series:
                                break
                    except (ValueError, IndexError):
                        continue

    print(f"Loaded {len(data)} time series")
    if len(data) > 0:
        lengths = [len(d) for d in data]
        print(
            f"Series lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}"
        )

    return data, names


def pad_series(data, target_length):
    """Pad time series to same length with zero padding at the beginning."""
    padded = []
    for series in data:
        if len(series) < target_length:
            pad_length = target_length - len(series)
            padded_series = np.concatenate(
                [np.zeros(pad_length, dtype=np.float32), series]
            )
        else:
            padded_series = series[-target_length:]
        padded.append(padded_series)
    return np.array(padded)


def create_sequences(data, seq_length, pred_horizon):
    """
    Create sliding window sequences for forecasting.

    Args:
        data: numpy array of shape (T,) - univariate time series
        seq_length: length of input sequence
        pred_horizon: how many steps ahead to predict

    Returns:
        X: input sequences (N, seq_length)
        y: target values (N, pred_horizon)
    """
    T = len(data)
    sequences = []
    targets = []

    for i in range(seq_length, T - pred_horizon + 1):
        seq = data[i - seq_length : i]  # (seq_length,)
        target = data[i : i + pred_horizon]  # (pred_horizon,)

        sequences.append(seq)
        targets.append(target)

    if len(sequences) == 0:
        return np.array([]).reshape(0, seq_length), np.array([]).reshape(
            0, pred_horizon
        )

    X = np.array(sequences)  # (N, seq_length)
    y = np.array(targets)  # (N, pred_horizon)
    return X, y


def create_multivariate_dataset(data_list, seq_length, pred_horizon):
    """
    Create multivariate sequences from multiple univariate time series.

    Args:
        data_list: list of univariate time series
        seq_length: length of input sequence
        pred_horizon: how many steps ahead to predict

    Returns:
        X: (N, seq_length, D) where D is number of features
        y: (N, pred_horizon, D)
    """
    # Pad all series to same length
    max_len = max(len(d) for d in data_list)
    print(f"Padding series to length {max_len}")
    data_array = pad_series(data_list, max_len)  # (D, T)

    # Transpose to (T, D)
    data_array = data_array.T  # (T, D)

    print(f"Data shape: {data_array.shape}")

    # Create sequences
    T, D = data_array.shape
    sequences = []
    targets = []

    for i in range(seq_length, T - pred_horizon + 1):
        seq = data_array[i - seq_length : i]  # (seq_length, D)
        target = data_array[i : i + pred_horizon]  # (pred_horizon, D)
        sequences.append(seq)
        targets.append(target)

    X = np.array(sequences, dtype=np.float32)  # (N, seq_length, D)
    y = np.array(targets, dtype=np.float32)  # (N, pred_horizon, D)

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
    y_train_flat = y_train.reshape(-1, y_train.shape[-1])
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
        "n_features": X_train.shape[-1],  # D dimension (number of series)
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
    print("Starting web traffic data preparation for time series forecasting...")

    # Load data
    data_list, names = load_tsf_file(
        DATA_PATH, num_series=N_SERIES, max_length=MAX_LENGTH
    )

    if len(data_list) == 0:
        print("ERROR: No data loaded!")
        return

    # Create multivariate sequences
    print("Creating multivariate sequences...")
    X, y = create_multivariate_dataset(data_list, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    print(f"Created {len(X)} sequences with shape X={X.shape}, y={y.shape}")

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
