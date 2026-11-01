# Training ulang model LSTM prediksi PM2.5 lewat MLflow Project di GitHub Actions (kriteria 3)
# Hyperparameter diambil dari trial terbaik random search di kriteria 2

import argparse
import os

import mlflow
import pandas as pd
import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beijing_air_quality_preprocessing")

BEST_PARAMS = {   # Hasil terbaik random search (trial_8, MAE test 9,30 µg/m³)
    "window_size": 48,
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dense_units": 8,
    "dropout": 0.0,
    "learning_rate": 0.0003,
    "batch_size": 256,
}


def make_windows(series, window_size, target_index):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    return ds.map(lambda w: (w[:-1], w[-1:, target_index]))   # Input beberapa jam ke belakang, target PM2.5 di jam berikutnya


def windowed_dataset(df, feature_columns, window_size, batch_size, target_index, shuffle_buffer=None):
    station_datasets = []
    for station in df["station"].unique():   # Window dibuat per stasiun biar nggak nyambung ke stasiun lain
        series = df[df["station"] == station][feature_columns].values.astype("float32")
        station_datasets.append(make_windows(series, window_size, target_index))

    ds = station_datasets[0]
    for other in station_datasets[1:]:
        ds = ds.concatenate(other)

    ds = ds.cache()
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer)   # Cuma data train yang diacak
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(p, n_features):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(p["window_size"], n_features)),
        tf.keras.layers.LSTM(p["lstm_units_1"], return_sequences=True, dropout=p["dropout"]),   # LSTM sebelum yang terakhir wajib return_sequences=True
        tf.keras.layers.LSTM(p["lstm_units_2"], dropout=p["dropout"]),
        tf.keras.layers.Dense(p["dense_units"], activation="relu"),
        tf.keras.layers.Dense(1),   # Prediksi PM2.5 satu jam ke depan
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
                  loss=tf.keras.losses.Huber(), metrics=["mae"])   # Huber lebih tahan sama lonjakan PM2.5
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)                   # Diisi dari parameter epochs di file MLproject
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    # Tracking URI dan experiment tidak diatur di sini, karena mlflow run sudah menyiapkan run-nya sendiri

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"), parse_dates=["datetime"])
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"), parse_dates=["datetime"])

    feature_columns = [col for col in train.columns if col not in ["station", "datetime"]]   # Semua kolom jadi fitur, kecuali penanda stasiun dan waktu
    target_index = feature_columns.index("PM2.5")

    p = BEST_PARAMS
    train_ds = windowed_dataset(train, feature_columns, p["window_size"], p["batch_size"], target_index, shuffle_buffer=20000)
    test_ds = windowed_dataset(test, feature_columns, p["window_size"], p["batch_size"], target_index)

    mlflow.tensorflow.autolog(log_input_examples=True)   # Parameter, metrik per epoch, dan model dicatat otomatis

    with mlflow.start_run():   # Melanjutkan run yang sudah dibuat oleh mlflow run
        model = build_model(p, len(feature_columns))
        model.fit(train_ds, validation_data=test_ds, epochs=args.epochs,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
