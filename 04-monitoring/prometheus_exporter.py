# Exporter Prometheus: meneruskan request ke model di container Docker dan mencatat metrik API, model, serta sistem
import os
import time

import joblib
import psutil
import requests
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:5005/invocations")   # Alamat model, bisa diarahkan ke Cloud Run lewat env MODEL_URL
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaler.pkl")

scaler = joblib.load(SCALER_PATH)                                             # Dipakai untuk mengembalikan prediksi ke satuan µg/m³
pm25_index = list(scaler.feature_names_in_).index("PM2.5")
PM25_MEAN, PM25_STD = scaler.mean_[pm25_index], scaler.scale_[pm25_index]

app = Flask(__name__)

# Metrik API
REQUEST_COUNT = Counter("http_requests_total", "Total request prediksi yang diterima")
REQUEST_FAILED = Counter("http_requests_failed_total", "Total request prediksi yang gagal")
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Waktu respons API dari exporter sampai model")
THROUGHPUT = Counter("http_requests_throughput", "Total request untuk menghitung request per detik")
IN_PROGRESS = Gauge("http_requests_in_progress", "Request yang sedang diproses")
PAYLOAD_WINDOWS = Gauge("model_request_windows", "Jumlah window (48 jam) dalam request terakhir")

# Metrik model
PREDICTION_LAST = Gauge("model_prediction_pm25_ugm3", "Prediksi PM2.5 terakhir dalam µg/m³")
PREDICTION_DIST = Histogram("model_prediction_pm25_distribution_ugm3", "Sebaran prediksi PM2.5 dalam µg/m³",
                            buckets=[12, 35, 55, 75, 115, 150, 250, 500])   # Batas kategori kualitas udara PM2.5
PREDICTION_TOTAL = Counter("model_predictions_total", "Total nilai prediksi yang dihasilkan model")

# Metrik sistem
CPU_USAGE = Gauge("system_cpu_usage", "Penggunaan CPU dalam persen")
RAM_USAGE = Gauge("system_ram_usage", "Penggunaan RAM dalam persen")
DISK_USAGE = Gauge("system_disk_usage", "Penggunaan disk dalam persen")
PROCESS_MEMORY = Gauge("exporter_process_memory_mb", "Memori yang dipakai proses exporter dalam MB")


@app.route("/metrics", methods=["GET"])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=1))                            # Diperbarui setiap kali Prometheus mengambil metrik
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage(os.path.abspath(os.sep)).percent)
    PROCESS_MEMORY.set(psutil.Process().memory_info().rss / 1024 ** 2)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    THROUGHPUT.inc()
    IN_PROGRESS.inc()

    data = request.get_json()
    PAYLOAD_WINDOWS.set(len(data.get("inputs", [])))

    try:
        response = requests.post(MODEL_URL, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        for value in result["predictions"]:                                   # Tiap window menghasilkan satu prediksi
            pm25 = float(value[0] if isinstance(value, list) else value) * PM25_STD + PM25_MEAN
            PREDICTION_LAST.set(pm25)
            PREDICTION_DIST.observe(pm25)
            PREDICTION_TOTAL.inc()

        return jsonify(result)

    except Exception as e:
        REQUEST_FAILED.inc()
        return jsonify({"error": str(e)}), 500

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)                   # Latensi dicatat untuk request sukses maupun gagal
        IN_PROGRESS.dec()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
