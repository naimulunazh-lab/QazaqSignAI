"""Train and export the browser classifier from a labelled landmark CSV.

Expected columns: gesture_id plus x0,y0,z0 ... x20,y20,z20.
Rows are single-hand landmark frames. Record many signers, angles and lighting conditions.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf


def feature_columns(frame: pd.DataFrame) -> list[str]:
    expected = [f"{axis}{point}" for point in range(21) for axis in "xyz"]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing landmark columns, e.g. {sorted(missing)[:3]}")
    return expected


def normalize(raw: np.ndarray) -> np.ndarray:
    """Center on wrist and normalize by the furthest landmark, matching web code."""
    hands = raw.reshape((-1, 21, 3)).astype("float32")
    hands -= hands[:, :1, :]
    scale = np.linalg.norm(hands, axis=2).max(axis=1, keepdims=True)
    return (hands / np.maximum(scale[:, :, None], 1e-6)).reshape((-1, 63))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../dataset/landmarks.csv")
    parser.add_argument("--out", default="../models/gesture-classifier")
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset).dropna()
    if "gesture_id" not in data:
        raise ValueError("CSV must contain gesture_id.")
    labels = sorted(data.gesture_id.astype(str).unique())
    if len(labels) < 2:
        raise ValueError("At least two gesture_id classes are required.")
    label_to_index = {label: index for index, label in enumerate(labels)}
    x = normalize(data[feature_columns(data)].to_numpy())
    y = np.array([label_to_index[label] for label in data.gesture_id.astype(str)])
    order = np.random.default_rng(42).permutation(len(x))
    x, y = x[order], y[order]
    split = max(1, int(.15 * len(x)))
    x_train, x_valid, y_train, y_valid = x[split:], x[:split], y[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input((63,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(len(labels), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=args.epochs, batch_size=32,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)], verbose=2)

    out = pathlib.Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    keras_path = out / "classifier.keras"; model.save(keras_path)
    with open(out / "labels.json", "w", encoding="utf-8") as file: json.dump(labels, file, ensure_ascii=False)
    # TensorFlow.js creates model.json + shard files directly consumed by TfjsModelAdapter.
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, str(out))
    print(f"Exported {len(labels)} labels and browser model to {out}")


if __name__ == "__main__":
    main()
