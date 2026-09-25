"""
extract_document.py
--------------------
SOURCE 3 of 3: Document/text ingestion (unstructured clinical feedback).

Format:            Unstructured plain text files (.txt)
Ingestion method:   Document file parser & NLP keyword extractor
Raw destination:    data/raw/document_raw/notes/
Intermediate zone:  data/raw/document_raw/specialty_feedback_notes.csv
Grain:              One clinical feedback document per medical specialty

Why this source adds value: The primary tabular encounter dataset has
no free-text clinical notes, nurse discharge documentation, or patient
feedback. In real-world healthcare analytics, unstructured clinician
notes and patient discharge comprehension reports provide critical signals
regarding post-discharge adherence risk.

Pipeline Role: This script ingests individual .txt documents from the
raw document landing area (`data/raw/document_raw/notes/`), extracts
structured metadata (Department, Document Type, Document ID), cleans
the free-text content, and computes an NLP keyword-based discharge risk score.
"""

import os
import re
import csv
import glob
import datetime
import logging

NOTES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "document_raw", "notes")
DOC_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "document_raw")
PROCESSED_NOTES_PATH = os.path.join(DOC_RAW_DIR, "specialty_feedback_notes.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "ingestion_log.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EXTRACT-DOC] %(message)s")
logger = logging.getLogger(__name__)

RISK_KEYWORDS = ["confus", "missed", "non-compliant", "difficulty", "barrier", "uncontrolled", "urgent"]


def _log_run(source_type, source_name, status, row_count, error_message=""):
    log_exists = os.path.exists(LOG_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["extraction_timestamp", "source_type", "source_name", "status", "row_count", "error_message"])
        writer.writerow([datetime.datetime.now().isoformat(), source_type, source_name, status, row_count, error_message])


def parse_document(file_path):
    """Parses an individual unstructured text note file and extracts metadata + content."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    dept_match = re.search(r"^DEPARTMENT:\s*(.+)$", raw_text, re.MULTILINE)
    doc_type_match = re.search(r"^DOCUMENT_TYPE:\s*(.+)$", raw_text, re.MULTILINE)
    doc_id_match = re.search(r"^DOCUMENT_ID:\s*(.+)$", raw_text, re.MULTILINE)
    content_match = re.search(r"CONTENT:\s*(.+)$", raw_text, re.MULTILINE | re.DOTALL)

    dept = dept_match.group(1).strip() if dept_match else "Unknown"
    doc_type = doc_type_match.group(1).strip() if doc_type_match else "General Note"
    doc_id = doc_id_match.group(1).strip() if doc_id_match else os.path.basename(file_path)
    content = content_match.group(1).strip() if content_match else raw_text.strip()

    # Calculate keyword risk score from unstructured text
    content_lower = content.lower()
    keyword_matches = [kw for kw in RISK_KEYWORDS if kw in content_lower]
    risk_score = len(keyword_matches)

    return {
        "medical_specialty": dept,
        "document_id": doc_id,
        "document_type": doc_type,
        "feedback_text": content.replace("\n", " ").strip(),
        "risk_keyword_score": risk_score,
        "notes_available": 1,
    }


def extract():
    status, error_message = "SUCCESS", ""
    records = []

    try:
        if not os.path.exists(NOTES_DIR):
            raise FileNotFoundError(f"Document raw landing directory not found: {NOTES_DIR}")

        file_paths = glob.glob(os.path.join(NOTES_DIR, "*.txt"))
        if not file_paths:
            raise FileNotFoundError(f"No document text files found in {NOTES_DIR}")

        for fp in file_paths:
            rec = parse_document(fp)
            records.append(rec)

        os.makedirs(DOC_RAW_DIR, exist_ok=True)
        fieldnames = ["medical_specialty", "document_id", "document_type", "feedback_text", "risk_keyword_score", "notes_available"]
        with open(PROCESSED_NOTES_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Ingested {len(records)} clinical text documents from {NOTES_DIR}")
    except Exception as e:
        status, error_message = "FAILED", str(e)
        logger.error(f"Document extraction failed: {error_message}")

    _log_run("document", "Clinical Discharge & Feedback Documents (raw TXT files)",
              status, len(records), error_message)

    if status == "FAILED":
        raise RuntimeError(error_message)

    return PROCESSED_NOTES_PATH, len(records)


if __name__ == "__main__":
    path, count = extract()
    print(f"[document_raw] Ingestion complete: {count} documents parsed -> {path}")
