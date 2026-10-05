"""
load_db.py
----------
Storage layer for Hospital Readmission Analytical Mart.

Loads the curated, integrated dataset across six tables:
1. admissions: Admission and discharge clinical categories and length of stay.
2. diagnoses: Enriched primary/secondary diagnoses with NLM clinical definitions.
3. patient_summary: Demographics, clinical procedures, and medications.
4. readmissions: Analytical fact table for readmission rates and risk cohorts.
5. diagnosis_reference: Raw Source 2 lookup reference table.
6. specialty_feedback: Raw Source 3 unstructured notes reference table.

Target backends:
- PostgreSQL (via DATABASE_URL environment variable)
- Local SQLite database (fallback at db/hospital_mart.db)
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

INTEGRATED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "staging", "integrated_encounters.csv")
API_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "api_raw", "icd9_lookup_cache.csv")
DOC_NOTES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "document_raw", "specialty_feedback_notes.csv")
SQLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "hospital_mart.db")
SCHEMA_SQL_PATH = os.path.join(os.path.dirname(__file__), "..", "sql", "schema.sql")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [LOAD] %(message)s")
logger = logging.getLogger(__name__)


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        logger.info("Connecting to PostgreSQL via DATABASE_URL")
        return create_engine(db_url), "postgresql"
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
    logger.info(f"DATABASE_URL not set; using local SQLite mart at {SQLITE_PATH}")
    return create_engine(f"sqlite:///{SQLITE_PATH}"), "sqlite"


def create_analytical_views(conn, backend):
    """Creates formal analytical views in the database mart for reporting and dashboards."""
    logger.info(f"Ensuring analytical data mart views exist ({backend})...")
    if backend == "postgresql":
        # Handled by schema.sql or CREATE OR REPLACE VIEW
        conn.execute(text("""
            CREATE OR REPLACE VIEW v_readmission_kpis AS
            SELECT
                COUNT(*) AS total_encounters,
                SUM(readmitted_30d) AS readmitted_30d_count,
                ROUND(AVG(readmitted_30d) * 100.0, 2) AS readmission_rate_pct,
                ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay_days,
                SUM(high_risk_flag) AS high_risk_cohort_count,
                ROUND(SUM(high_risk_flag) * 100.0 / COUNT(*), 2) AS high_risk_pct,
                SUM(feedback_available) AS encounters_with_feedback
            FROM readmissions;
        """))
        conn.execute(text("""
            CREATE OR REPLACE VIEW v_high_risk_cohort AS
            SELECT
                r.surrogate_key,
                r.age_group,
                r.medical_specialty,
                r.admission_type_name,
                r.discharge_disposition_name,
                r.length_of_stay,
                r.diagnosis_count,
                r.prior_admissions,
                r.readmitted_30d,
                d.diag_1_description,
                r.risk_keyword_score
            FROM readmissions r
            LEFT JOIN diagnoses d ON r.surrogate_key = d.surrogate_key
            WHERE r.high_risk_flag = 1;
        """))
    else:
        conn.execute(text("DROP VIEW IF EXISTS v_readmission_kpis;"))
        conn.execute(text("""
            CREATE VIEW v_readmission_kpis AS
            SELECT
                COUNT(*) AS total_encounters,
                SUM(readmitted_30d) AS readmitted_30d_count,
                ROUND(AVG(readmitted_30d) * 100.0, 2) AS readmission_rate_pct,
                ROUND(AVG(length_of_stay), 2) AS avg_length_of_stay_days,
                SUM(high_risk_flag) AS high_risk_cohort_count,
                ROUND(SUM(high_risk_flag) * 100.0 / COUNT(*), 2) AS high_risk_pct,
                SUM(feedback_available) AS encounters_with_feedback
            FROM readmissions;
        """))
        conn.execute(text("DROP VIEW IF EXISTS v_high_risk_cohort;"))
        conn.execute(text("""
            CREATE VIEW v_high_risk_cohort AS
            SELECT
                r.surrogate_key,
                r.age_group,
                r.medical_specialty,
                r.admission_type_name,
                r.discharge_disposition_name,
                r.length_of_stay,
                r.diagnosis_count,
                r.prior_admissions,
                r.readmitted_30d,
                d.diag_1_description,
                r.risk_keyword_score
            FROM readmissions r
            LEFT JOIN diagnoses d ON r.surrogate_key = d.surrogate_key
            WHERE r.high_risk_flag = 1;
        """))
    conn.commit()


def load():
    if not os.path.exists(INTEGRATED_PATH):
        raise FileNotFoundError(f"Integrated dataset not found at {INTEGRATED_PATH}")

    df = pd.read_csv(INTEGRATED_PATH, low_memory=False)
    api_lookup = pd.read_csv(API_CACHE_PATH)
    doc_notes = pd.read_csv(DOC_NOTES_PATH)
    engine, backend = get_engine()

    # Prepare tabular subsets
    admissions_cols = [
        "surrogate_key", "admission_type_id", "admission_type_name",
        "admission_source_id", "admission_source_name",
        "discharge_disposition_id", "discharge_disposition_name",
        "length_of_stay", "medical_specialty", "payer_code",
    ]
    admissions = df[[c for c in admissions_cols if c in df.columns]].copy()

    diag_cols = [
        "surrogate_key", "diag_1", "diag_2", "diag_3", "diagnosis_count",
        "diag_1_description", "diag_1_lookup_status",
    ]
    diagnoses = df[[c for c in diag_cols if c in df.columns]].copy()

    patient_cols = [
        "surrogate_key", "race", "gender", "age_group", "num_lab_procedures",
        "num_procedures", "medication_count", "prior_admissions",
        "high_risk_flag", "has_insulin", "change_med", "has_diabetes_med",
    ]
    patient_summary = df[[c for c in patient_cols if c in df.columns]].copy()

    readm_cols = [
        "surrogate_key", "readmitted", "readmitted_30d", "age_group",
        "medical_specialty", "admission_type_name", "discharge_disposition_name",
        "length_of_stay", "diagnosis_count", "prior_admissions",
        "high_risk_flag", "risk_keyword_score", "feedback_available",
    ]
    readmissions = df[[c for c in readm_cols if c in df.columns]].copy()

    if backend == "postgresql":
        logger.info("Initializing PostgreSQL relational schema and constraints...")
        with engine.connect() as conn:
            if os.path.exists(SCHEMA_SQL_PATH):
                with open(SCHEMA_SQL_PATH, "r", encoding="utf-8") as sf:
                    schema_ddl = sf.read()
                # Execute DDL statements
                for stmt in schema_ddl.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
                conn.commit()
                logger.info("PostgreSQL schema DDL executed successfully.")

        # Insert rows using append to respect foreign keys and constraints
        admissions.to_sql("admissions", engine, if_exists="append", index=False, chunksize=10000)
        diagnoses.to_sql("diagnoses", engine, if_exists="append", index=False, chunksize=10000)
        patient_summary.to_sql("patient_summary", engine, if_exists="append", index=False, chunksize=10000)
        readmissions.to_sql("readmissions", engine, if_exists="append", index=False, chunksize=10000)
        api_lookup.to_sql("diagnosis_reference", engine, if_exists="append", index=False)
        doc_notes.to_sql("specialty_feedback", engine, if_exists="append", index=False)
    else:
        # SQLite storage path
        admissions.to_sql("admissions", engine, if_exists="replace", index=False)
        diagnoses.to_sql("diagnoses", engine, if_exists="replace", index=False)
        patient_summary.to_sql("patient_summary", engine, if_exists="replace", index=False)
        readmissions.to_sql("readmissions", engine, if_exists="replace", index=False)
        api_lookup.to_sql("diagnosis_reference", engine, if_exists="replace", index=False)
        doc_notes.to_sql("specialty_feedback", engine, if_exists="replace", index=False)

        # Create indexes for analytical performance
        with engine.connect() as conn:
            try:
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_adm_surrogate ON admissions(surrogate_key);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_diag_surrogate ON diagnoses(surrogate_key);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_readm_surrogate ON readmissions(surrogate_key);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_readm_target ON readmissions(readmitted_30d);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_readm_age ON readmissions(age_group);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_readm_risk ON readmissions(high_risk_flag);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_adm_type ON admissions(admission_type_name);"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dsch_disp ON admissions(discharge_disposition_name);"))
                conn.commit()
            except Exception as e:
                logger.warning(f"Index creation note: {e}")

    # Ensure analytical views exist
    with engine.connect() as conn:
        create_analytical_views(conn, backend)

    # Verify table counts and view integrity
    counts = {}
    with engine.connect() as conn:
        for table in ["admissions", "diagnoses", "patient_summary", "readmissions",
                      "diagnosis_reference", "specialty_feedback"]:
            counts[table] = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

        counts["v_readmission_kpis (view)"] = conn.execute(text("SELECT COUNT(*) FROM v_readmission_kpis")).scalar()
        counts["v_high_risk_cohort (view)"] = conn.execute(text("SELECT COUNT(*) FROM v_high_risk_cohort")).scalar()

    for item, count in counts.items():
        logger.info(f"Loaded / Verified {count} records in '{item}' ({backend})")

    target = os.environ.get("DATABASE_URL", SQLITE_PATH)
    return target, counts


if __name__ == "__main__":
    path, table_counts = load()
    print(f"Load complete -> {path}")
    print(table_counts)
