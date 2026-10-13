"""
pipeline_dag.py
---------------
Orchestration layer for Hospital Readmission Analytics & Clinical MLOps Pipeline.

Dependency Graph:
    extract_file_task ------\\
    extract_api_task    -----+--> transform_task --> integrate_task --> load_task
    extract_document_task --/                                               │
                                                                            ▼ (Optional / With MLOps)
                                                                     ml_features_task
                                                                            │
                                                                            ▼
                                                                       ml_train_task
                                                                            │
                                                                            ▼
                                                                      ml_explain_task
                                                                            │
                                                                            ▼
                                                                      ml_monitor_task

Supports two modes:
  1. Default: Full Part 1 Data Engineering ETL (6 tasks)
  2. Full MLOps: Part 1 ETL + Part 2 Machine Learning Operations (10 tasks via --all or --mlops)
"""

import os
import sys
import json
import logging
import datetime
import argparse
import traceback

# Ensure src is on sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extract_file import extract as extract_file
from extract_api import extract as extract_api
from extract_document import extract as extract_document
from transform import transform
from integrate import integrate
from load_db import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DAG] %(message)s")
logger = logging.getLogger(__name__)

RUN_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "pipeline_runs.json")


def run_task(name, func, max_retries=1):
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        try:
            logger.info(f"TASK START: {name} (attempt {attempt})")
            result = func()
            logger.info(f"TASK SUCCESS: {name}")
            return {"task": name, "status": "SUCCESS", "attempt": attempt, "result": str(result)}
        except Exception as e:
            logger.error(f"TASK FAILED: {name} on attempt {attempt}: {e}")
            if attempt > max_retries:
                return {
                    "task": name,
                    "status": "FAILED",
                    "attempt": attempt,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }


def run_pipeline(with_mlops: bool = False):
    run_start = datetime.datetime.now().isoformat()
    results = []

    def ok(r):
        return r and r.get("status") == "SUCCESS"

    # --- Part 1: Independent extraction tasks (3 segregated sources) ---
    r_file = run_task("extract_file_task", extract_file)
    results.append(r_file)
    r_api = run_task("extract_api_task", extract_api)
    results.append(r_api)
    r_doc = run_task("extract_document_task", extract_document)
    results.append(r_doc)

    # --- Transform (depends on file source only) ---
    if ok(r_file):
        r_transform = run_task("transform_task", transform)
    else:
        r_transform = {"task": "transform_task", "status": "SKIPPED"}
    results.append(r_transform)

    # --- Integrate (depends on ALL THREE sources + transform) ---
    if ok(r_transform) and ok(r_api) and ok(r_doc):
        r_integrate = run_task("integrate_task", integrate)
    else:
        r_integrate = {"task": "integrate_task", "status": "SKIPPED"}
    results.append(r_integrate)

    # --- Load Data Mart (depends on integrate) ---
    if ok(r_integrate):
        r_load = run_task("load_task", load)
    else:
        r_load = {"task": "load_task", "status": "SKIPPED"}
    results.append(r_load)

    # --- Part 2: MLOps Pipeline Extension ---
    if with_mlops:
        if ok(r_load):
            from src.ml_features import run_feature_pipeline
            r_ml_feat = run_task("ml_features_task", run_feature_pipeline)
        else:
            r_ml_feat = {"task": "ml_features_task", "status": "SKIPPED"}
        results.append(r_ml_feat)

        if ok(r_ml_feat):
            from src.ml_train import run_training_pipeline
            r_ml_train = run_task("ml_train_task", run_training_pipeline)
        else:
            r_ml_train = {"task": "ml_train_task", "status": "SKIPPED"}
        results.append(r_ml_train)

        if ok(r_ml_train):
            from src.ml_explain import run_explainability_pipeline
            r_ml_explain = run_task("ml_explain_task", run_explainability_pipeline)
        else:
            r_ml_explain = {"task": "ml_explain_task", "status": "SKIPPED"}
        results.append(r_ml_explain)

        if ok(r_ml_explain):
            from src.ml_monitor import run_monitoring_audit
            r_ml_mon = run_task("ml_monitor_task", run_monitoring_audit)
        else:
            r_ml_mon = {"task": "ml_monitor_task", "status": "SKIPPED"}
        results.append(r_ml_mon)

    run_end = datetime.datetime.now().isoformat()
    overall_status = "SUCCESS" if all(t["status"] == "SUCCESS" for t in results) else "FAILED"

    run_record = {
        "run_start": run_start,
        "run_end": run_end,
        "mode": "ETL + MLOps" if with_mlops else "ETL Only",
        "overall_status": overall_status,
        "tasks": results,
    }

    os.makedirs(os.path.dirname(RUN_LOG_PATH), exist_ok=True)
    history = []
    if os.path.exists(RUN_LOG_PATH):
        with open(RUN_LOG_PATH, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    history.append(run_record)
    with open(RUN_LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"PIPELINE RUN COMPLETED: {overall_status} (Mode: {run_record['mode']})")
    return run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare Pipeline Orchestrator")
    parser.add_argument("--mlops", "--all", dest="with_mlops", action="store_true", help="Run full pipeline including MLOps stages")
    args = parser.parse_args()

    record = run_pipeline(with_mlops=args.with_mlops)
    print(f"Summary: Status={record['overall_status']}, Tasks={len(record['tasks'])}")
