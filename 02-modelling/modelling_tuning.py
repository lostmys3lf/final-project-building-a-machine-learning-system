# Random search hyperparameter LSTM, dicatat manual ke MLflow di DagsHub (kriteria 2 Advanced)
# Model terbaik didaftarkan ke Model Registry

import argparse
import os
import random

import dagshub
import matplotlib.pyplot as plt
import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from modelling import DATA_DIR, build_model, windowed_dataset   # Fungsi windowing dan arsitektur dipakai bareng dengan modelling.py

SEARCH_SPACE = {
    "window_size": [12, 24, 48],
    "lstm_units_1": [32, 64, 128],
    "lstm_units_2": [16, 32, 64],
    "dense_units": [8, 16, 32],
    "dropout": [0.0, 0.1, 0.2],
    "learning_rate": [0.001, 0.0005, 0.0003],
}
BATCH_SIZE = 256


def log_artifacts(model, history, y_true, y_pred, metrics, input_example):
    mlflow.tensorflow.log_model(model, "model", input_example=input_example)   # Artefak model, signature dibentuk dari input example
    mlflow.log_dict(metrics, "metric_info.json")                                # Ringkasan metrik

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))                             # Grafik loss dan MAE per epoch
    axes[0].plot(history.history["loss"], label="train"); axes[0].plot(history.history["val_loss"], label="test"); axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["mae"], label="train"); axes[1].plot(history.history["val_mae"], label="test"); axes[1].set_title("MAE"); axes[1].legend()
    mlflow.log_figure(fig, "loss_dan_mae.png"); plt.close(fig)

    fig = plt.figure(figsize=(14, 5))                                           # Grafik prediksi vs asli
    plt.plot(y_true[:500], label="Asli"); plt.plot(y_pred[:500], label="Prediksi")
    plt.title("PM2.5 asli vs prediksi (500 jam pertama di test)"); plt.legend()
    mlflow.log_figure(fig, "prediksi_vs_asli.png"); plt.close(fig)

    summary_lines = []                                                          # Ringkasan arsitektur model
    model.summary(print_fn=lambda line, **kwargs: summary_lines.append(line))
    mlflow.log_text("\n".join(summary_lines), "model_summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    dagshub.init(repo_owner="lostmys3lf", repo_name="pm25-lstm-model", mlflow=True)   # Semua run langsung tercatat di DagsHub
    mlflow.set_experiment("Prediksi PM2.5 LSTM")

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"), parse_dates=["datetime"])
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"), parse_dates=["datetime"])
    scaler = joblib.load(os.path.join(args.data_dir, "scaler.pkl"))

    feature_columns = [col for col in train.columns if col not in ["station", "datetime"]]
    target_index = feature_columns.index("PM2.5")
    pm25_index = list(scaler.feature_names_in_).index("PM2.5")
    pm25_mean, pm25_std = scaler.mean_[pm25_index], scaler.scale_[pm25_index]   # Buat ngembaliin prediksi ke µg/m³

    dataset_cache = {}   # Window per ukuran disimpan, biar nggak dibuat ulang tiap trial
    random.seed(42)
    results = []

    for trial in range(args.n_trials):
        p = {name: random.choice(choices) for name, choices in SEARCH_SPACE.items()}
        p["batch_size"] = BATCH_SIZE

        if p["window_size"] not in dataset_cache:
            dataset_cache[p["window_size"]] = (
                windowed_dataset(train, feature_columns, p["window_size"], BATCH_SIZE, target_index, shuffle_buffer=20000),
                windowed_dataset(test, feature_columns, p["window_size"], BATCH_SIZE, target_index),
            )
        train_ds, test_ds = dataset_cache[p["window_size"]]

        with mlflow.start_run(run_name=f"trial_{trial + 1}") as run:
            mlflow.log_params(p)
            model = build_model(p, len(feature_columns))
            history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, verbose=2,
                                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])

            for epoch in range(len(history.history["loss"])):   # Metrik per epoch, sama seperti yang dicatat autolog
                mlflow.log_metrics({name: values[epoch] for name, values in history.history.items()}, step=epoch)

            y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel() * pm25_std + pm25_mean
            y_pred = model.predict(test_ds, verbose=0).ravel() * pm25_std + pm25_mean
            y_last_hour = np.concatenate([x.numpy()[:, -1, target_index] for x, _ in test_ds]) * pm25_std + pm25_mean   # Tebakan "sama kayak jam sebelumnya"

            metrics = {
                "test_mae_ugm3": float(np.mean(np.abs(y_true - y_pred))),
                "test_rmse_ugm3": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "baseline_mae_ugm3": float(np.mean(np.abs(y_true - y_last_hour))),
            }
            mlflow.log_metrics(metrics)

            input_example = next(iter(train_ds))[0][0:5].numpy()   # 5 sampel pertama, sama polanya dengan X_train[0:5] di materi
            log_artifacts(model, history, y_true, y_pred, metrics, input_example)

            results.append({"run_id": run.info.run_id, **metrics, **p})
            print(f"Trial {trial + 1}: MAE {metrics['test_mae_ugm3']:.2f} µg/m³ | baseline {metrics['baseline_mae_ugm3']:.2f}")

    hasil_search = pd.DataFrame(results).sort_values("test_mae_ugm3")
    print(hasil_search.to_string(index=False))

    best = hasil_search.iloc[0]   # Registry model dengan MAE test paling kecil
    print("Run terbaik:", best["run_id"], "| MAE:", round(best["test_mae_ugm3"], 2), "µg/m³")
    mlflow.register_model(f"runs:/{best['run_id']}/model", "prediksi-pm25-lstm")
