"""
extract_file.py
----------------
SOURCE 1 of 3: File ingestion (structured, batch).

Format:            CSV
Ingestion method:   File ingestion (Python/Pandas)
Raw destination:    data/raw/file_raw/
Grain:              One row per hospital encounter (patient visit)

This is the primary, encounter-level source: the UCI "Diabetes 130-US
Hospitals" dataset. It provides the core structured record -- admission,
diagnosis codes, procedures, medications, and the readmission label --
that the other two sources (API, document) are joined onto.
"""

import os
import csv
import datetime
import logging

RAW_SOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "file_raw", "diabetic_data.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "ingestion_log.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EXTRACT-FILE] %(message)s")
logger = logging.getLogger(__name__)


def _log_run(source_type, source_name, status, row_count, error_message=""):
    log_exists = os.path.exists(LOG_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["extraction_timestamp", "source_type", "source_name", "status", "row_count", "error_message"])
        writer.writerow([datetime.datetime.now().isoformat(), source_type, source_name, status, row_count, error_message])


def extract():
    status, row_count, error_message = "SUCCESS", 0, ""
    try:
        if not os.path.exists(RAW_SOURCE_PATH):
            raise FileNotFoundError(f"Raw source file not found at {RAW_SOURCE_PATH}")
        with open(RAW_SOURCE_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header
            row_count = sum(1 for _ in reader)
        logger.info(f"Extracted {row_count} rows from {RAW_SOURCE_PATH}")
    except Exception as e:
        status, error_message = "FAILED", str(e)
        logger.error(f"Extraction failed: {error_message}")

    _log_run("file", "UCI Diabetes 130-US Hospitals (diabetic_data.csv)", status, row_count, error_message)

    if status == "FAILED":
        raise RuntimeError(error_message)
    return RAW_SOURCE_PATH, row_count


if __name__ == "__main__":
    path, rows = extract()
    print(f"[file_raw] Extraction complete: {rows} rows -> {path}")
