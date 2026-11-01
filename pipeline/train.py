# Continuous training: latih ulang LSTM dengan hyperparameter terbaik, catat manual ke MLflow di DagsHub
# Arsitektur, windowing, dan artefak memakai kode yang sama dengan 02-modelling (experimental-operational symmetry)

import argparse
import json
import os
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "02-modelling"))
from modelling import BEST_PARAMS, build_model, windowed_dataset   # noqa: E402
from modelling_tuning import log_artifacts                          # noqa: E402

EXPERIMENT_NAME = "Prediksi PM2.5 LSTM - Continuous Training"


def load_data(data_dir, window_size, batch_size):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), parse_dates=["datetime"])
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), parse_dates=["datetime"])
    scaler = joblib.load(os.path.join(data_dir, "scaler.pkl"))

    feature_columns = [col for col in train.columns if col not in ["station", "datetime"]]
    target_index = feature_columns.index("PM2.5")
    pm25_index = list(scaler.feature_names_in_).index("PM2.5")
    train_ds = windowed_dataset(train, feature_columns, window_size, batch_size, target_index, shuffle_buffer=20000)
    test_ds = windowed_dataset(test, feature_columns, window_size, batch_size, target_index)
    return train_ds, test_ds, feature_columns, target_index, scaler.mean_[pm25_index], scaler.scale_[pm25_index]


def evaluate(model, test_ds, target_index, pm25_mean, pm25_std):
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel() * pm25_std + pm25_mean
    y_pred = model.predict(test_ds, verbose=0).ravel() * pm25_std + pm25_mean
    y_last_hour = np.concatenate([x.numpy()[:, -1, target_index] for x, _ in test_ds]) * pm25_std + pm25_mean   # Baseline "sama kayak jam sebelumnya"
    metrics = {
        "test_mae_ugm3": float(np.mean(np.abs(y_true - y_pred))),
        "test_rmse_ugm3": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "baseline_mae_ugm3": float(np.mean(np.abs(y_true - y_last_hour))),
    }
    return y_true, y_pred, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--data_report", default="reports/data_validation.json")
    parser.add_argument("--output", default="reports/candidate.json")
    args = parser.parse_args()

    p = dict(BEST_PARAMS)
    train_ds, test_ds, feature_columns, target_index, pm25_mean, pm25_std = load_data(args.data_dir, p["window_size"], p["batch_size"])

    mlflow.set_experiment(EXPERIMENT_NAME)                                     # Tracking URI dan kredensial DagsHub dari environment variable
    with mlflow.start_run(run_name=f"ct-{os.environ.get('GITHUB_RUN_NUMBER', 'local')}") as run:
        mlflow.set_tags({"pipeline": "continuous-training",
                         "git_sha": os.environ.get("GITHUB_SHA", "local"),
                         "trigger": os.environ.get("GITHUB_EVENT_NAME", "manual")})   # Metadata asal run untuk lineage
        mlflow.log_params({**p, "epochs": args.epochs, "n_features": len(feature_columns)})
        if os.path.exists(args.data_report):
            mlflow.log_artifact(args.data_report)                              # Hasil data validation ikut tersimpan

        model = build_model(p, len(feature_columns))
        history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, verbose=2,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metrics({name: values[epoch] for name, values in history.history.items()}, step=epoch)

        y_true, y_pred, metrics = evaluate(model, test_ds, target_index, pm25_mean, pm25_std)
        mlflow.log_metrics(metrics)
        input_example = next(iter(train_ds))[0][0:5].numpy()
        log_artifacts(model, history, y_true, y_pred, metrics, input_example)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        json.dump({"run_id": run.info.run_id, **metrics}, open(args.output, "w"), indent=2)
        print(f"Run {run.info.run_id}: MAE {metrics['test_mae_ugm3']:.2f} µg/m³ | baseline {metrics['baseline_mae_ugm3']:.2f}")
