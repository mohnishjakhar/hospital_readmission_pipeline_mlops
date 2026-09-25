"""
pipeline_dag.py
---------------
Orchestration layer, extended for 3 segregated sources.

DAG shape (dependency order):

    extract_file_task ------\\
    extract_api_task    -----+--> transform_task --> integrate_task --> load_task
    extract_document_task --/

extract_file_task, extract_api_task, and extract_document_task have no
dependency on each other (each hits a different, independent source),
but transform_task only cleans the file source, so it only strictly
needs extract_file_task to have succeeded. integrate_task needs ALL
THREE extraction outputs plus the cleaned staging file, so it is
gated on all four upstream tasks.

As before: no full Airflow install was used (see README); this
runner reproduces Airflow's core DAG semantics -- named tasks,
explicit dependency order, per-task logging, retry-on-failure, and a
persisted run history.
"""

import os
import json
import logging
import datetime
import traceback

from extract_file import extract as extract_file
from extract_api import extract as extract_api
from extract_document import extract as extract_document
from transform import transform
from integrate import integrate
from load_db import load

logging.basicConfig(level=logging.INFO, format="%(asctime)s [DAG] %(message)s")
logger = logging.getLogger(__name__)

RUN_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "pipeline_runs.json")


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
                return {"task": name, "status": "FAILED", "attempt": attempt,
                         "error": str(e), "traceback": traceback.format_exc()}


def run_pipeline():
    run_start = datetime.datetime.now().isoformat()
    results = []

    def ok(r):
        return r["status"] == "SUCCESS"

    # --- Independent extraction tasks (3 segregated sources) ---
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

    # --- Load (depends on integrate) ---
    if ok(r_integrate):
        r_load = run_task("load_task", load)
    else:
        r_load = {"task": "load_task", "status": "SKIPPED"}
    results.append(r_load)

    run_end = datetime.datetime.now().isoformat()
    overall_status = "SUCCESS" if all(t["status"] == "SUCCESS" for t in results) else "FAILED"

    run_record = {
        "run_start": run_start, "run_end": run_end,
        "overall_status": overall_status, "tasks": results,
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

    logger.info(f"PIPELINE RUN {overall_status}")
    return run_record


if __name__ == "__main__":
    record = run_pipeline()
    print(json.dumps(record, indent=2))
