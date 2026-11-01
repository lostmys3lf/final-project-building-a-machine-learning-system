# Otomatisasi preprocessing dataset Beijing Multi-Site Air Quality
# Langkahnya sama dengan notebook eksperimen, cuma dipecah jadi fungsi supaya bisa dijalankan ulang otomatis

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

WIND_DIRECTION_DEGREES = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
                          "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}

TEST_START_DATE = "2016-03-01"   # 12 bulan terakhir jadi data test, sama seperti di notebook


def load_data(raw_dir):
    file_paths = sorted(glob.glob(os.path.join(raw_dir, "**", "PRSA_Data_*.csv"), recursive=True))   # Ambil 12 file CSV stasiun saja
    if not file_paths:
        raise FileNotFoundError(f"File CSV stasiun tidak ditemukan di {raw_dir}")
    return pd.concat([pd.read_csv(path) for path in file_paths], ignore_index=True)   # Gabung semua stasiun jadi satu DataFrame


def clean_datetime(df):
    df = df.copy()                                                           # Kerja di salinan, data asli tetap utuh
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])    # Gabung kolom waktu jadi satu
    df = df.drop(columns=["No", "year", "month", "day", "hour"])              # No cuma nomor baris, kolom waktu udah diwakili datetime
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)       # Urut per stasiun lalu per jam
    return df.drop_duplicates(subset=["station", "datetime"])                 # Satu stasiun nggak boleh punya dua data di jam yang sama


def handle_outlier(df):
    df = df.copy()
    df.loc[df["O3"] == 1071, "O3"] = np.nan   # 1071 muncul belasan jam berturut-turut di Dongsi, tanda sensor nyangkut, jadi dianggap missing value
    return df


def fill_missing_values(df):
    df = df.copy()
    df[NUMERIC_COLUMNS] = df.groupby("station")[NUMERIC_COLUMNS].transform(
        lambda col: col.interpolate(method="linear").ffill().bfill())   # Interpolasi per stasiun, ffill dan bfill buat jam kosong di awal atau akhir data
    df["wd"] = df.groupby("station")["wd"].transform(
        lambda col: col.ffill().bfill())                                 # Arah angin berupa teks, jadi diisi pakai arah di jam sebelumnya
    return df


def encode_features(df):
    df = df.copy()
    radians = np.deg2rad(df["wd"].map(WIND_DIRECTION_DEGREES))   # Ubah arah angin ke sudut dulu
    df["wd_sin"] = np.sin(radians)                               # Arah angin itu melingkar, jadi dipecah ke sin dan cos
    df["wd_cos"] = np.cos(radians)
    df = df.drop(columns=["wd"])
    return pd.concat([df, pd.get_dummies(df["station"], prefix="st", dtype=int)], axis=1)   # One-hot stasiun, kolom station aslinya tetap disimpan


def split_and_scale(df):
    test_start = pd.Timestamp(TEST_START_DATE)
    train = df[df["datetime"] < test_start].copy()
    test = df[df["datetime"] >= test_start].copy()

    scaler = StandardScaler()
    train[NUMERIC_COLUMNS] = scaler.fit_transform(train[NUMERIC_COLUMNS])   # fit cuma di train, biar data test nggak bocor
    test[NUMERIC_COLUMNS] = scaler.transform(test[NUMERIC_COLUMNS])
    return train, test, scaler


def save_output(train, test, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False, float_format="%.4f")   # 4 angka desimal biar ukuran file aman buat GitHub
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False, float_format="%.4f")
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))                          # Dipakai lagi buat ngembaliin prediksi PM2.5 ke satuan asli


def preprocess_data(raw_dir, output_dir):
    df = load_data(raw_dir)
    df = clean_datetime(df)
    df = handle_outlier(df)
    df = fill_missing_values(df)
    df = encode_features(df)
    train, test, scaler = split_and_scale(df)
    save_output(train, test, scaler, output_dir)
    return train, test   # Data yang siap dilatih


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))   # Path dihitung dari lokasi file ini, jadi aman dijalankan dari folder mana pun

    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset Beijing Multi-Site Air Quality")
    parser.add_argument("--raw", default=os.path.join(script_dir, "..", "beijing_air_quality_raw"),
                        help="Folder dataset mentah")
    parser.add_argument("--output", default=os.path.join(script_dir, "beijing_air_quality_preprocessing"),
                        help="Folder hasil preprocessing")
    args = parser.parse_args()

    train, test = preprocess_data(args.raw, args.output)
    print("Train:", train.shape, "| Test:", test.shape)
    print("Sisa missing value:", train.isnull().sum().sum() + test.isnull().sum().sum())   # Harusnya 0
