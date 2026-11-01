# Data validation sebelum training (MLOps level 1)
# Skema yang tidak cocok menghentikan pipeline; pergeseran distribusi dicatat sebagai drift

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd


def write_github_output(**values):
    path = os.environ.get("GITHUB_OUTPUT")                                   # Dipakai step berikutnya di GitHub Actions
    if path:
        with open(path, "a") as f:
            for key, value in values.items():
                f.write(f"{key}={value}\n")


def check_schema(name, df, schema, errors):
    if list(df.columns) != schema["columns"]:
        errors.append(f"{name}: urutan atau nama kolom berbeda dari skema")
        return
    feature_columns = [c for c in schema["columns"] if c not in schema["id_columns"]]
    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"{name}: kolom bukan numerik {non_numeric}")
        return
    values = df[feature_columns].to_numpy(dtype="float64")
    if np.isnan(values).any() or np.isinf(values).any():                    # Model tidak bisa dilatih dengan NaN atau inf
        errors.append(f"{name}: masih ada nilai NaN atau inf")
    if set(df["station"].unique()) != set(schema["stations"]):
        errors.append(f"{name}: daftar stasiun berbeda dari skema")
    if len(df) < schema["min_rows"][name]:
        errors.append(f"{name}: jumlah baris {len(df)} di bawah minimum {schema['min_rows'][name]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--schema", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_schema.json"))
    parser.add_argument("--report", default="reports/data_validation.json")
    args = parser.parse_args()

    schema = json.load(open(args.schema))
    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"), parse_dates=["datetime"])
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"), parse_dates=["datetime"])
    scaler = joblib.load(os.path.join(args.data_dir, "scaler.pkl"))

    errors = []
    check_schema("train", train, schema, errors)
    check_schema("test", test, schema, errors)
    test_start = pd.Timestamp(schema["test_start_date"])
    if train["datetime"].max() >= test_start or test["datetime"].min() < test_start:   # Cegah data test bocor ke train
        errors.append("Pembagian train/test tidak sesuai tanggal batas")

    drift = {}
    for column, mean, std in zip(scaler.feature_names_in_, scaler.mean_, scaler.scale_):   # Statistik data mentah sebelum scaling
        ref = schema["reference_stats"][column]
        smd = abs(mean - ref["mean"]) / ref["std"]                           # Standardized mean difference terhadap data referensi
        drift[column] = {"mean": round(float(mean), 4), "reference_mean": ref["mean"], "smd": round(float(smd), 4),
                         "std_ratio": round(float(std / ref["std"]), 4)}
    drifted = [c for c, d in drift.items() if d["smd"] > schema["drift_threshold_smd"]]

    report = {"status": "failed" if errors else "passed", "schema_errors": errors,
              "rows": {"train": len(train), "test": len(test)}, "drift_detected": bool(drifted),
              "drifted_columns": drifted, "drift": drift}
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=2)
    write_github_output(drift_detected=str(bool(drifted)).lower())

    print(json.dumps({k: report[k] for k in ["status", "schema_errors", "rows", "drifted_columns"]}, indent=2))
    if errors:
        sys.exit("Data validation gagal: skema tidak cocok, training dibatalkan")   # Schema skew menghentikan pipeline
