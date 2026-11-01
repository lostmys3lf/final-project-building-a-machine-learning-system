# Ingest data: unduh dataset Beijing Multi-Site Air Quality terbaru dari UCI, lalu ekstrak sampai ketemu CSV per stasiun

import argparse
import glob
import io
import os
import urllib.request
import zipfile

DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",          # Link unduhan halaman dataset UCI
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip",   # Link lama sebagai cadangan
]


def download(urls):
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                print("Data diunduh dari", url)
                return response.read()
        except Exception as error:                                            # Coba link berikutnya kalau gagal
            print("Gagal mengunduh", url, "->", error)
    raise SystemExit("Semua sumber data gagal diunduh, pipeline dihentikan")


def extract_nested_zip(zip_bytes, output_dir):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for name in archive.namelist():
            if name.lower().endswith(".zip"):                                 # Dataset UCI berisi zip di dalam zip
                extract_nested_zip(archive.read(name), output_dir)
            elif name.lower().endswith(".csv"):
                archive.extract(name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unduh dan ekstrak dataset mentah dari UCI")
    parser.add_argument("--output", default="data/raw", help="Folder tujuan CSV mentah")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    extract_nested_zip(download(DATASET_URLS), args.output)                   # Selalu ambil versi terbaru dari sumber

    csv_files = glob.glob(os.path.join(args.output, "**", "PRSA_Data_*.csv"), recursive=True)
    if not csv_files:
        raise SystemExit("CSV stasiun tidak ditemukan setelah ekstrak, pipeline dihentikan")
    print(f"{len(csv_files)} file CSV stasiun siap di {args.output}")
