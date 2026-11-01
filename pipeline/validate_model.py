# Model validation (MLOps level 1): model kandidat dibandingkan dengan model champion di data test yang sama
# Kandidat dipromosikan kalau mengalahkan baseline dan tidak lebih buruk dari champion melebihi toleransi (non-regression)

import argparse
import json
import os

import mlflow
from mlflow import MlflowClient

from train import evaluate, load_data
from modelling import BEST_PARAMS


def write_github_output(**values):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            for key, value in values.items():
                f.write(f"{key}={value}\n")


def write_summary(lines):
    path = os.environ.get("GITHUB_STEP_SUMMARY")                             # Ringkasan tampil di halaman run GitHub Actions
    if path:
        with open(path, "a") as f:
            f.write("\n".join(lines) + "\n")


def champion_mae(client, model_name, alias, data_dir):
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return None, None, "belum ada champion"
    try:                                                                     # Champion dievaluasi ulang di data test saat ini
        p = dict(BEST_PARAMS)
        _, test_ds, _, target_index, pm25_mean, pm25_std = load_data(data_dir, p["window_size"], p["batch_size"])
        model = mlflow.tensorflow.load_model(f"models:/{model_name}@{alias}")
        _, _, metrics = evaluate(model, test_ds, target_index, pm25_mean, pm25_std)
        return version, metrics["test_mae_ugm3"], "evaluasi ulang di data test saat ini"
    except Exception as error:                                               # Cadangan kalau model lama gagal dimuat
        print("Champion gagal dimuat, pakai metrik yang tercatat:", error)
        logged = client.get_run(version.run_id).data.metrics["test_mae_ugm3"]
        return version, logged, "metrik tercatat di registry"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="reports/candidate.json")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_name", default="prediksi-pm25-lstm")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--report", default="reports/model_validation.json")
    parser.add_argument("--tolerance", type=float, default=0.01)              # Batas penurunan akurasi yang masih diterima (1%)
    args = parser.parse_args()

    client = MlflowClient()
    candidate = json.load(open(args.candidate))
    version, champ_mae, method = champion_mae(client, args.model_name, args.alias, args.data_dir)

    checks = {"beats_baseline": candidate["test_mae_ugm3"] < candidate["baseline_mae_ugm3"],
              "no_regression_vs_champion": champ_mae is None or candidate["test_mae_ugm3"] <= champ_mae * (1 + args.tolerance)}   # Model hasil pipeline dengan data terbaru diutamakan selama tidak turun signifikan
    promote = all(checks.values())

    report = {"candidate_run_id": candidate["run_id"], "candidate_mae_ugm3": candidate["test_mae_ugm3"],
              "baseline_mae_ugm3": candidate["baseline_mae_ugm3"],
              "champion_version": version.version if version else None, "champion_mae_ugm3": champ_mae,
              "champion_eval_method": method, "tolerance": args.tolerance, "checks": checks, "promote": promote}

    new_version = ""
    if promote:
        registered = mlflow.register_model(f"runs:/{candidate['run_id']}/model", args.model_name)
        client.set_registered_model_alias(args.model_name, args.alias, registered.version)   # Alias champion pindah ke versi baru
        new_version = registered.version
        report["new_champion_version"] = new_version

    client.set_tag(candidate["run_id"], "model_validation", "passed" if promote else "rejected")
    if champ_mae is not None:
        client.log_metric(candidate["run_id"], "champion_mae_ugm3", champ_mae)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=2)
    client.log_artifact(candidate["run_id"], args.report)

    champ_text = f"{champ_mae:.2f}" if champ_mae is not None else "-"
    write_summary(["## Model validation",
                   "| Model | MAE (µg/m³) |", "|---|---|",
                   f"| Kandidat | {candidate['test_mae_ugm3']:.2f} |",
                   f"| Champion v{report['champion_version']} ({method}) | {champ_text} |",
                   f"| Baseline jam sebelumnya | {candidate['baseline_mae_ugm3']:.2f} |",
                   f"\nToleransi non-regression: {args.tolerance:.0%} dari MAE champion",
                   f"\n**Keputusan:** {'promote ke champion v' + str(new_version) if promote else 'champion dipertahankan'}"])
    write_github_output(promote=str(promote).lower(), run_id=candidate["run_id"], model_version=new_version)
    print(json.dumps(report, indent=2))
