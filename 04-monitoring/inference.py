# Kirim request prediksi PM2.5 ke exporter, lalu tampilkan hasilnya dalam µg/m³
import argparse
import os
import random
import time

import joblib
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"                                     # Lewat exporter supaya metrik tercatat
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_SIZE = 48                                                             # Sama dengan window_size model terbaik


def load_test_data(test_path):
    test = pd.read_csv(test_path, parse_dates=["datetime"])
    feature_columns = [col for col in test.columns if col not in ["station", "datetime"]]   # Urutan kolom sama dengan saat training
    return test, feature_columns


def make_windows(test, feature_columns, station, start, n_windows):
    rows = test[test["station"] == station].reset_index(drop=True)
    inputs, actual = [], []
    for i in range(start, start + n_windows):
        inputs.append(rows.loc[i:i + WINDOW_SIZE - 1, feature_columns].values.tolist())   # 48 jam ke belakang
        actual.append(rows.loc[i + WINDOW_SIZE, "PM2.5"])                                # PM2.5 di jam berikutnya
    return inputs, actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", default=os.path.join(BASE_DIR, "..", "01-experiment", "beijing_air_quality_preprocessing", "test.csv"))
    parser.add_argument("--n_requests", type=int, default=20)
    args = parser.parse_args()

    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    pm25_index = list(scaler.feature_names_in_).index("PM2.5")
    pm25_mean, pm25_std = scaler.mean_[pm25_index], scaler.scale_[pm25_index]

    test, feature_columns = load_test_data(args.test_path)
    stations = test["station"].unique().tolist()

    for n in range(args.n_requests):
        station = random.choice(stations)
        start = random.randint(0, 2000)
        inputs, actual = make_windows(test, feature_columns, station, start, n_windows=random.randint(1, 4))

        response = requests.post(API_URL, json={"inputs": inputs}, timeout=30)
        if response.status_code != 200:
            print(f"Request {n + 1}: error {response.status_code} {response.text}")
            continue

        predictions = response.json()["predictions"]
        for pred, real in zip(predictions, actual):
            pred_value = (pred[0] if isinstance(pred, list) else pred) * pm25_std + pm25_mean
            real_value = real * pm25_std + pm25_mean
            print(f"Request {n + 1} | {station}: prediksi {pred_value:.1f} µg/m³ | asli {real_value:.1f} µg/m³")
        time.sleep(0.5)                                                      # Jeda kecil supaya grafik di Grafana terlihat bertahap
