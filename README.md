# Hospital Readmission Analytics & Prediction (Multi-Source Pipeline)

**Individual Project**: Data Engineering & MLOps  
**Part 1 Deliverable**: Multi-Source Healthcare Data Pipeline, Analytical PostgreSQL Mart, and Interactive Streamlit Analytics Dashboard.  
**Compliance**: Fully aligned with **Project 5 Specification** and **Student Guideline on Segregated Multi-Source Data Engineering**.

---

## 1. Multi-Source Architecture (Strict Raw Segregation)

In accordance with the project guidelines, data is ingested across **three genuinely different, segregated source types** without blind merging:

| # | Source Name | Source Type | Raw Format | Ingestion Method | Raw Destination Landing Zone | Ingestion Role |
|---|---|---|---|---|---|---|
| **1** | UCI Diabetes 130-US Hospitals | Structured File | CSV (101,766 rows) | File Batch Ingestion (`pandas`) | `data/raw/file_raw/diabetic_data.csv` | Primary inpatient encounter spine (demographics, labs, meds, outcomes) |
| **2** | NLM Clinical Tables Search Service | Public REST API | JSON via HTTP GET | Concurrent API Connector (`requests` + `ThreadPoolExecutor`) | `data/raw/api_raw/icd9_lookup_cache.csv` | Resolves opaque ICD-9 codes to clinical terminology (*Congestive heart failure*, etc.) |
| **3** | Clinical Discharge Summaries & Notes | Unstructured Documents | Plain Text (.txt files) | Document Ingestion & NLP Parser (`glob` + `re`) | `data/raw/document_raw/notes/*.txt` | Ingests qualitative clinical notes and extracts discharge adherence risk scores |

Detailed justifications, join cardinality, and missing-data lineage indicators are documented in [`docs/data_dictionary.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/docs/data_dictionary.md).

```
  Source 1: File (CSV)    -----\
  Source 2: REST API (JSON) ----+--> Validation & ETL --> Entity Integration --> Analytical Mart --> Streamlit Dashboard
  Source 3: Documents (TXT) ---/      (Cleaning, Hash,     (Status Flags,        (PostgreSQL /      (5 Core Views +
                                        ID Mappings)       NULL Lineage)          SQLite)            Interactive Filters)
```

---

## 2. Orchestration: Airflow DAG & Standalone Runner

The pipeline supports two execution modalities:
1. **Production Apache Airflow 2.x DAG** ([`dags/hospital_readmission_dag.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/dags/hospital_readmission_dag.py)):
   - Declares tasks via `PythonOperator` with retry policies and explicit DAG dependencies:
     `[extract_file_task, extract_api_task, extract_document_task] >> transform_task >> integrate_task >> load_db_task`
2. **Lightweight Standalone DAG Runner** ([`src/pipeline_dag.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/pipeline_dag.py)):
   - Executes the exact same DAG dependency order with retry logic and logs run history to `logs/pipeline_runs.json` with zero server overhead.

---

## 3. Database Layer: PostgreSQL & SQLite Dual Engine

- **PostgreSQL (Recommended)**:
  Run the included Docker Compose service to launch PostgreSQL 15 and Adminer with one command:
  ```bash
  docker compose up -d
  ```
  Set the database environment variable:
  ```powershell
  $env:DATABASE_URL="postgresql+psycopg2://postgres:hospital_secure_password123@localhost:5432/hospital_mart"
  ```
- **SQLite (Zero-Setup Fallback)**:
  If `DATABASE_URL` is not set, the pipeline automatically writes to and queries `db/hospital_mart.db`.
- **Production DDL**:
  Formal SQL table schemas, primary/foreign key constraints, B-Tree indexes, and analytical reporting views are defined in [`sql/schema.sql`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/sql/schema.sql).

---

## 4. Quick Start & Execution

### 4.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4.2 Run Full End-to-End Pipeline
```bash
python src/pipeline_dag.py
```
*Execution summary:*
- Ingests 101,766 raw encounters from Source 1.
- Ingests top 60 diagnosis descriptions concurrently from Source 2 NLM REST API.
- Parses 14 unstructured clinical discharge documents from Source 3.
- Quarantines 2,423 expired/hospice cases and 29,353 repeat visits into `data/rejected/rejected_records.csv`.
- Maps admission, discharge, and source numeric codes using `IDs_mapping.csv`.
- Integrates all 3 sources into `data/staging/integrated_encounters.csv` (51,959 NLM API matches; missing note scores preserved as `NULL`).
- Loads 69,990 clean curated records into 6 tables in the data mart.

### 4.3 Launch the Analytics Dashboard
```bash
streamlit run dashboard/app.py
```
*Dashboard features:*
- Executive KPI cards with baseline comparisons.
- Interactive multi-parameter filters (Medical Specialty, Age Bracket, Admission Type, Risk Tier).
- **View 1**: Readmission Rate by Age Group.
- **View 2**: Readmission by Diagnosis (Enriched via NLM REST API).
- **View 3**: Length of Stay Distribution (Readmitted vs Non-Readmitted).
- **View 4**: Admission & Discharge Clinical Patterns (Emergency/Elective + Discharge Destinations).
- **View 5**: High-Risk Cohort Intelligence Dashboard (2.21x Risk Multiplier cohort).
- **View 6**: Multi-Source Qualitative Document Signal (Adherence risk score correlation).

---

## 5. Repository Structure

```
hospital_readmission_pipeline1.0/
├── dags/
│   └── hospital_readmission_dag.py             # Official Apache Airflow 2.x DAG
├── data/
│   ├── raw/
│   │   ├── file_raw/
│   │   │   ├── diabetic_data.csv               # Source 1: Raw UCI encounter CSV
│   │   │   └── IDs_mapping.csv                 # UCI admission/discharge code mappings
│   │   ├── api_raw/
│   │   │   └── icd9_lookup_cache.csv           # Source 2: NLM REST API cache
│   │   └── document_raw/
│   │       ├── notes/*.txt                     # Source 3: Unstructured clinical text files
│   │       └── specialty_feedback_notes.csv    # Parsed document staging table
│   ├── staging/
│   │   ├── encounters_clean.csv                # Cleaned Source 1 staging dataset
│   │   └── integrated_encounters.csv           # Curated, ML-ready 3-source dataset
│   └── rejected/
│       └── rejected_records.csv                # Quarantined expired & repeat encounters
├── db/
│   └── hospital_mart.db                        # SQLite analytical data mart
├── sql/
│   └── schema.sql                              # PostgreSQL production DDL schema & views
├── dashboard/
│   └── app.py                                  # Interactive Streamlit analytics dashboard
├── logs/
│   ├── ingestion_log.csv                       # Per-source extraction metadata & timestamps
│   ├── integration_log.csv                     # Entity matching & missing-data quality metrics
│   └── pipeline_runs.json                      # Orchestration execution history
├── src/
│   ├── extract_file.py                         # Source 1 batch file ingestion
│   ├── extract_api.py                          # Source 2 concurrent REST API ingestion
│   ├── extract_document.py                     # Source 3 unstructured document parser
│   ├── transform.py                            # Data cleaning, ID mapping, & feature engineering
│   ├── integrate.py                            # Multi-source entity matching & quality flags
│   ├── load_db.py                              # Relational mart loader (PostgreSQL/SQLite)
│   └── pipeline_dag.py                         # Standalone Python DAG orchestrator
├── docs/
│   ├── data_dictionary.md                      # Comprehensive data dictionary & validation rules
│   ├── Project5_Part1_Comprehensive_Report.md  # 8-10 page academic technical report
│   ├── Project5_Part1_Report.docx              # Formatted Word document report
│   └── architecture_diagram.png                # High-resolution pipeline architecture diagram
├── docker-compose.yml                          # PostgreSQL 15 & Adminer container setup
├── requirements.txt                            # Python environment dependencies
└── README.md                                   # Project documentation & run guide
```

---

## 6. Part 1 Submission Package Checklist (10 Items)

| Item # | Required Submission Deliverable | Location in Repository | Status |
|---|---|---|---|
| **1** | Source code and ETL scripts | [`src/extract_file.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/extract_file.py), [`extract_api.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/extract_api.py), [`extract_document.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/extract_document.py), [`transform.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/transform.py), [`integrate.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/integrate.py) | **Complete** |
| **2** | Airflow DAG or approved orchestration workflow | [`dags/hospital_readmission_dag.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/dags/hospital_readmission_dag.py) and [`src/pipeline_dag.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/pipeline_dag.py) | **Complete** |
| **3** | Database schema and sample populated tables | [`sql/schema.sql`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/sql/schema.sql) and [`db/hospital_mart.db`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/db/hospital_mart.db) | **Complete** |
| **4** | Dataset source information and access instructions | Documented in [`docs/data_dictionary.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/docs/data_dictionary.md) and [`README.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/README.md) | **Complete** |
| **5** | Architecture diagram and pipeline flow | [`docs/architecture_diagram.png`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/docs/architecture_diagram.png) and Excalidraw design | **Complete** |
| **6** | Data dictionary and validation rules | [`docs/data_dictionary.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/docs/data_dictionary.md) | **Complete** |
| **7** | Streamlit analytical dashboard application | [`dashboard/app.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/dashboard/app.py) (5 core views + filters) | **Complete** |
| **8** | Project report of approximately 8–10 pages | [`docs/Project5_Part1_Comprehensive_Report.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/docs/Project5_Part1_Comprehensive_Report.md) (3,700+ words) & `.docx` | **Complete** |
| **9** | Execution evidence & screenshots | Log files in [`logs/`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/logs/) and chart images in [`docs/`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/docs/) | **Complete** |
| **10**| README with setup and run instructions | [`README.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/README.md) | **Complete** |
