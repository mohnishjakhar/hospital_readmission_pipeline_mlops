"""
integrate.py
------------
Entity matching & integration layer across 3 segregated data sources:

1. encounters_clean.csv (Source 1: file_raw)
2. icd9_lookup_cache.csv (Source 2: api_raw)
3. specialty_feedback_notes.csv (Source 3: document_raw)

Join Keys & Mapping Strategy:
- Source 1 <-> Source 2: Joined on primary diagnosis code (diag_1 == icd9_code).
  Enriches encounter with authoritative clinical diagnosis descriptions from NLM API.
- Source 1 <-> Source 3: Joined on medical_specialty.
  Enriches encounter with qualitative discharge notes and NLP-derived risk scores.

Handling Incomplete Records (per Student Guideline Section 9):
- Missing join results are NEVER silently filled with zeros or assumed to be successful.
- Traceability columns are created:
  * diag_1_lookup_status: MATCHED, FALLBACK_OFFLINE, NOT_FOUND, or NOT_ATTEMPTED.
  * feedback_available: 1 if note exists, 0 if missing.
  * risk_keyword_score: Populated if feedback_available=1, otherwise genuinely NULL (pd.NA).
"""

import os
import csv
import logging
import pandas as pd

STAGING_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "staging")
CLEANED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")
ENCOUNTERS_CLEAN_PATH = os.path.join(STAGING_DIR, "encounters_clean.csv")
API_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "api_raw", "icd9_lookup_cache.csv")
DOC_NOTES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "document_raw", "specialty_feedback_notes.csv")
INTEGRATED_PATH = os.path.join(STAGING_DIR, "integrated_encounters.csv")
CLEANED_INTEGRATED_PATH = os.path.join(CLEANED_DIR, "integrated_encounters.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "integration_log.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INTEGRATE] %(message)s")
logger = logging.getLogger(__name__)


def integrate():
    os.makedirs(STAGING_DIR, exist_ok=True)
    os.makedirs(CLEANED_DIR, exist_ok=True)

    if not os.path.exists(ENCOUNTERS_CLEAN_PATH):
        raise FileNotFoundError(f"Cleaned encounters not found at {ENCOUNTERS_CLEAN_PATH}")
    if not os.path.exists(API_CACHE_PATH):
        raise FileNotFoundError(f"API lookup cache not found at {API_CACHE_PATH}")
    if not os.path.exists(DOC_NOTES_PATH):
        raise FileNotFoundError(f"Document notes not found at {DOC_NOTES_PATH}")

    encounters = pd.read_csv(ENCOUNTERS_CLEAN_PATH, low_memory=False)
    api_lookup = pd.read_csv(API_CACHE_PATH)
    doc_notes = pd.read_csv(DOC_NOTES_PATH)

    # 1. Join encounters with Source 2 (API lookup) on diag_1
    api_sub = api_lookup[["icd9_code", "description", "lookup_status"]].rename(
        columns={
            "icd9_code": "diag_1",
            "description": "diag_1_description",
            "lookup_status": "diag_1_lookup_status",
        }
    )
    # Ensure join key types match as strings
    encounters["diag_1"] = encounters["diag_1"].astype(str)
    api_sub["diag_1"] = api_sub["diag_1"].astype(str)

    merged = encounters.merge(api_sub, on="diag_1", how="left")
    # Codes outside our top-N query scope were never attempted; mark explicitly
    merged["diag_1_lookup_status"] = merged["diag_1_lookup_status"].fillna("NOT_ATTEMPTED")

    # 2. Join with Source 3 (Document ingestion) on medical_specialty
    doc_sub = doc_notes[["medical_specialty", "document_id", "document_type", "feedback_text", "risk_keyword_score", "notes_available"]]
    merged = merged.merge(doc_sub, on="medical_specialty", how="left")

    merged["feedback_available"] = merged["notes_available"].fillna(0).astype(int)
    # Ensure missing feedback risk score is preserved as NA (NOT zero)
    merged.loc[merged["feedback_available"] == 0, "risk_keyword_score"] = pd.NA
    merged.drop(columns=["notes_available"], inplace=True)

    # Save integrated analytical dataset (both staging and formal cleaned layers)
    merged.to_csv(INTEGRATED_PATH, index=False)
    merged.to_csv(CLEANED_INTEGRATED_PATH, index=False)

    # 3. Log integration quality metrics
    diag_status_counts = merged["diag_1_lookup_status"].value_counts().to_dict()
    feedback_available_count = int(merged["feedback_available"].sum())
    feedback_missing_count = int((merged["feedback_available"] == 0).sum())

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_integrated_rows", len(merged)])
        for status, count in diag_status_counts.items():
            writer.writerow([f"diag_1_lookup_status={status}", count])
        writer.writerow(["feedback_available=1", feedback_available_count])
        writer.writerow(["feedback_available=0", feedback_missing_count])

    logger.info(f"Integration complete: {len(merged)} rows curated.")
    logger.info(f"Diagnosis lookup status: {diag_status_counts}")
    logger.info(f"Document notes matched: {feedback_available_count} / missing: {feedback_missing_count}")

    return INTEGRATED_PATH, len(merged)


if __name__ == "__main__":
    path, count = integrate()
    print(f"Integration complete: {count} rows -> {path}")
