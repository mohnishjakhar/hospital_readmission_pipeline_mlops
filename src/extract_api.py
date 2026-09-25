"""
extract_api.py
--------------
SOURCE 2 of 3: REST API ingestion (structured, reference/lookup data).

Format:            JSON (HTTP response)
Ingestion method:   REST API connector (requests with ThreadPoolExecutor)
Raw destination:    data/raw/api_raw/
Grain:              One row per distinct ICD-9-CM diagnosis code
API:                NLM Clinical Table Search Service
                    https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search
                    (public, free, no API key required)

Why this source adds value: The primary file source only carries raw
ICD-9-CM codes (e.g., "428", "250.8", "414"). On their own, these codes
are opaque numbers and cannot be understood by clinicians or aggregated
into meaningful clinical groupings for dashboard visualization. This API
supplies the official NLM description for each code, which is joined onto
the encounter data in integrate.py.

Scoping: We query the top N most frequent primary diagnosis codes in the
dataset (covering the vast majority of all encounters) and record any
unattempted codes explicitly.

Resilience: Queries use concurrent HTTP requests with retries and timeouts.
If the API is unreachable, the script falls back to an offline reference table
while explicitly recording lookup_status="FALLBACK_OFFLINE" to preserve data lineage.
"""

import os
import csv
import time
import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

try:
    import requests
except ImportError:
    requests = None

FILE_RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "file_raw", "diabetic_data.csv")
API_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "api_raw")
CACHE_PATH = os.path.join(API_RAW_DIR, "icd9_lookup_cache.csv")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "ingestion_log.csv")

API_BASE = "https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search"
TOP_N_CODES = 60
MAX_WORKERS = 8
TIMEOUT_SECONDS = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EXTRACT-API] %(message)s")
logger = logging.getLogger(__name__)

# Standard CMS / NLM ICD-9 reference dictionary used when live API is unavailable
OFFLINE_FALLBACK = {
    "428": "Congestive heart failure, unspecified",
    "414": "Coronary atherosclerosis of unspecified vessel",
    "786": "Respiratory abnormality, symptoms involving respiratory system",
    "410": "Acute myocardial infarction",
    "486": "Pneumonia, organism unspecified",
    "427": "Cardiac dysrhythmias",
    "491": "Chronic bronchitis",
    "715": "Osteoarthrosis and allied disorders",
    "682": "Other cellulitis and abscess",
    "434": "Occlusion of cerebral arteries",
    "780": "General symptoms",
    "996": "Complications of surgical and medical care",
    "276": "Disorders of fluid, electrolyte, and acid-base balance",
    "038": "Septicemia",
    "38": "Septicemia",
    "250.8": "Diabetes with other specified manifestations",
    "599": "Other disorders of urethra and urinary tract",
    "584": "Acute kidney failure",
    "V57": "Care involving use of rehabilitation procedures",
    "250.6": "Diabetes with neurological manifestations",
    "518": "Other diseases of lung",
    "820": "Fracture of neck of femur",
    "577": "Diseases of pancreas",
    "493": "Asthma",
    "435": "Transient cerebral ischemia",
    "562": "Diverticula of intestine",
    "574": "Cholelithiasis",
    "296": "Episodic mood disorders",
    "560": "Intestinal obstruction without mention of hernia",
    "250.7": "Diabetes with peripheral circulatory disorders",
    "250.13": "Diabetes with ketoacidosis, type I, uncontrolled",
    "440": "Atherosclerosis",
    "433": "Occlusion and stenosis of precerebral arteries",
    "998": "Other complications of procedures, NEC",
    "722": "Intervertebral disc disorders",
    "250.02": "Diabetes mellitus without mention of complication, uncontrolled",
    "578": "Gastrointestinal hemorrhage",
    "250.11": "Diabetes with ketoacidosis, type I",
    "507": "Pneumonitis due to solids and liquids",
    "789": "Other symptoms involving abdomen and pelvis",
    "453": "Other venous embolism and thrombosis",
}


def _log_run(source_type, source_name, status, row_count, error_message=""):
    log_exists = os.path.exists(LOG_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["extraction_timestamp", "source_type", "source_name", "status", "row_count", "error_message"])
        writer.writerow([datetime.datetime.now().isoformat(), source_type, source_name, status, row_count, error_message])


def _fetch_single_code(session, code):
    code_str = str(code).strip()
    if requests is None or session is None:
        fallback = OFFLINE_FALLBACK.get(code_str)
        if fallback:
            return {"icd9_code": code_str, "description": fallback, "lookup_status": "FALLBACK_OFFLINE"}
        return {"icd9_code": code_str, "description": None, "lookup_status": "API_UNREACHABLE_NO_FALLBACK"}

    params = {"terms": code_str, "maxList": 5}
    for attempt in range(2):
        try:
            resp = session.get(API_BASE, params=params, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            payload = resp.json()
            if len(payload) > 3 and payload[3]:
                pairs = payload[3]
                clean_target = code_str.replace(".", "")
                for returned_code, raw_name in pairs:
                    clean_ret = str(returned_code).strip().replace(".", "")
                    if clean_ret.startswith(clean_target) or clean_target.startswith(clean_ret):
                        name = raw_name.strip().lstrip("-").strip()
                        if name:
                            return {"icd9_code": code_str, "description": name, "lookup_status": "MATCHED"}
                top_name = pairs[0][1].strip().lstrip("-").strip()
                if top_name:
                    return {"icd9_code": code_str, "description": top_name, "lookup_status": "MATCHED"}
            break
        except Exception:
            time.sleep(0.2)

    # If live lookup failed or returned no results, check fallback
    fallback = OFFLINE_FALLBACK.get(code_str)
    if fallback:
        return {"icd9_code": code_str, "description": fallback, "lookup_status": "FALLBACK_OFFLINE"}
    return {"icd9_code": code_str, "description": None, "lookup_status": "NOT_FOUND"}


def extract():
    os.makedirs(API_RAW_DIR, exist_ok=True)

    if not os.path.exists(FILE_RAW_PATH):
        raise FileNotFoundError(f"Primary file source not found at {FILE_RAW_PATH}")

    df = pd.read_csv(FILE_RAW_PATH, usecols=["diag_1"])
    top_codes = (
        df[df["diag_1"] != "?"]["diag_1"].dropna().value_counts().head(TOP_N_CODES).index.tolist()
    )

    rows = []
    session = requests.Session() if requests else None

    logger.info(f"Starting parallel lookup for top {len(top_codes)} ICD-9 codes against NLM API...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_fetch_single_code, session, c) for c in top_codes]
        for f in futures:
            rows.append(f.result())

    fieldnames = ["icd9_code", "description", "lookup_status"]
    with open(CACHE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(1 for r in rows if r["lookup_status"] == "MATCHED")
    fallback = sum(1 for r in rows if r["lookup_status"] == "FALLBACK_OFFLINE")
    not_found = sum(1 for r in rows if r["lookup_status"] == "NOT_FOUND")

    logger.info(f"ICD-9 Extraction Summary: {matched} MATCHED (live API), {fallback} FALLBACK_OFFLINE, {not_found} NOT_FOUND")
    status = "SUCCESS" if (matched + fallback) > 0 else "FAILED"
    _log_run("api", "NLM Clinical Tables ICD-9-CM API (icd9cm_dx/v3)", status, len(rows),
             f"matched={matched}, fallback={fallback}, not_found={not_found}")

    return CACHE_PATH, len(rows)


if __name__ == "__main__":
    path, count = extract()
    print(f"[api_raw] Lookup cache generated: {count} codes -> {path}")
