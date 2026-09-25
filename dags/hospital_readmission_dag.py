"""
hospital_readmission_dag.py
----------------------------
Official Apache Airflow 2.x DAG for Hospital Readmission Analytics Pipeline.

Implements multi-source data ingestion, validation, integration, and storage
across three segregated healthcare data sources:
1. Primary Hospital Encounter CSV (file_raw)
2. NLM Clinical Tables ICD-9-CM REST API (api_raw)
3. Clinical Discharge & Feedback Unstructured Documents (document_raw)

DAG Dependency Graph:
    extract_file_task     ---\\
    extract_api_task      ----+--> transform_task --> integrate_task --> load_db_task
    extract_document_task ---/
"""

import sys
import os
from datetime import datetime, timedelta

# Ensure src/ modules are importable in Airflow worker context
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_extract_file(**kwargs):
    from extract_file import extract
    path, count = extract()
    return {"raw_path": path, "row_count": count}


def task_extract_api(**kwargs):
    from extract_api import extract
    path, count = extract()
    return {"cache_path": path, "code_count": count}


def task_extract_document(**kwargs):
    from extract_document import extract
    path, count = extract()
    return {"doc_path": path, "doc_count": count}


def task_transform(**kwargs):
    from transform import transform
    path, kept, rejected = transform()
    return {"clean_path": path, "clean_count": kept, "rejected_count": rejected}


def task_integrate(**kwargs):
    from integrate import integrate
    path, count = integrate()
    return {"integrated_path": path, "total_count": count}


def task_load_db(**kwargs):
    from load_db import load
    target, counts = load()
    return {"target": target, "table_counts": counts}


default_args = {
    "owner": "hospital_data_engineering",
    "depends_on_past": False,
    "start_date": datetime(2026, 9, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="hospital_readmission_pipeline",
        default_args=default_args,
        description="Multi-source healthcare ETL pipeline for hospital readmission analytics",
        schedule_interval="@daily",
        catchup=False,
        tags=["healthcare", "data_engineering", "multi_source", "readmission"],
    ) as dag:

        t_extract_file = PythonOperator(
            task_id="extract_file_task",
            python_callable=task_extract_file,
            doc_md="Ingest primary UCI Diabetes hospital encounter CSV into data/raw/file_raw/",
        )

        t_extract_api = PythonOperator(
            task_id="extract_api_task",
            python_callable=task_extract_api,
            doc_md="Ingest ICD-9 clinical definitions from NLM REST API into data/raw/api_raw/",
        )

        t_extract_doc = PythonOperator(
            task_id="extract_document_task",
            python_callable=task_extract_document,
            doc_md="Ingest and parse clinical feedback notes from data/raw/document_raw/notes/",
        )

        t_transform = PythonOperator(
            task_id="transform_task",
            python_callable=task_transform,
            doc_md="Clean encounters, remove expired/repeat records, map IDs, and engineer features",
        )

        t_integrate = PythonOperator(
            task_id="integrate_task",
            python_callable=task_integrate,
            doc_md="Entity matching & join across all 3 segregated sources with missing-data flags",
        )

        t_load_db = PythonOperator(
            task_id="load_db_task",
            python_callable=task_load_db,
            doc_md="Load curated dataset into analytical tables and data marts in PostgreSQL/SQLite",
        )

        # Explicit DAG dependencies
        t_extract_file >> t_transform
        [t_transform, t_extract_api, t_extract_doc] >> t_integrate >> t_load_db


if __name__ == "__main__":
    if AIRFLOW_AVAILABLE:
        print("Testing DAG locally via Airflow 2.x dag.test()...")
        dag.test()
    else:
        print("Apache Airflow is not installed in the local Python environment.")
        print("Executing pipeline via standalone DAG runner (src/pipeline_dag.py)...")
        from pipeline_dag import run_pipeline
        run_pipeline()
