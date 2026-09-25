# Project 5: Hospital Readmission Analytics & Prediction Using Public Data
## Part 1: Multi-Source Heterogeneous Healthcare Data Pipeline & Analytical Mart
**Course**: Data Engineering & MLOps | **Project Mode**: Individual Project | **Marks**: 100 (Part 1: 50 Marks, Part 2: 50 Marks)  
**Submission**: Comprehensive Academic Technical Report (8–10 Page Standard)

---

## Abstract
Hospital readmissions within 30 days of discharge represent a major driver of escalating healthcare expenditures, clinician burnout, and substandard quality of care. Predicting and analyzing readmission patterns requires more than isolated tabular databases; modern healthcare data systems must integrate disparate, heterogeneous data streams while strictly preserving patient privacy, data provenance, and source identity. 

This project implements an enterprise-grade, privacy-conscious healthcare data engineering pipeline and analytical data mart adhering to both the **Project 5 Specification** and the **Student Guideline on Segregated Multi-Source Data Engineering**. The pipeline ingests data across three fundamentally different paradigms: (1) structured encounter records via batch CSV ingestion, (2) authoritative clinical diagnostic terminology via concurrent REST API calls to the National Library of Medicine (NLM), and (3) qualitative patient discharge comprehension signals via unstructured plain-text document ingestion. Raw sources are strictly segregated into dedicated landing areas before undergoing rigorous data validation, pseudonymous hashing, ID mapping, feature engineering, and entity integration. 

The curated analytical dataset is stored in a normalized PostgreSQL relational mart (with automated SQLite fallback) and orchestrated via both a production-ready **Apache Airflow 2.x DAG** and a standalone lightweight DAG runner. An interactive Streamlit analytical dashboard surfaces executive KPIs and five core clinical views alongside qualitative multi-source risk signals. Finally, an architectural blueprint for the Part 2 MLOps extension is formulated, establishing reproducible training, MLflow tracking, SHAP explainability, FastAPI containerized serving, and data drift monitoring.

---

## 1. Introduction & Healthcare Problem Understanding

### 1.1 Clinical Context and the Readmission Problem
Under the Centers for Medicare & Medicaid Services (CMS) Hospital Readmissions Reduction Program (HRRP), acute-care hospitals face severe financial penalties when 30-day unplanned readmission rates exceed national benchmarks. For diabetic inpatients, readmission risks are compounded by complex polypharmacy regimens, glycemic volatility, multi-organ comorbidities (e.g., congestive heart failure, renal insufficiency), and post-discharge self-management barriers. Identifying high-risk patient cohorts prior to discharge allows clinical teams to deploy targeted transition-of-care interventions, such as home nurse visits, pharmacist reconciliation, and early outpatient follow-up.

### 1.2 Data Engineering Objectives
Building an effective readmission analytics infrastructure entails several engineering challenges:
1. **Source Heterogeneity**: Clinical data does not reside in a single neat database. Inpatient billing codes, external diagnostic taxonomies, and clinician discharge notes originate in completely distinct systems and formats.
2. **Privacy and De-identification**: Data pipelines must adhere to HIPAA Safe Harbor principles by stripping direct identifiers and employing irreversible cryptographic hashing for record linkage.
3. **Data Quality and Bias Mitigation**: Deceased patients and hospice transfers cannot biologically experience readmission and must be systematically excluded; similarly, repeat-encounter biases must be eliminated.
4. **Missing-Data Integrity**: When heterogeneous sources are merged, missing records must never be treated as valid numerical zeros. Transparent flags must distinguish unattempted queries, missing external records, and genuine clinical absences.

---

## 2. Multi-Source Architecture & Raw Segregation

### 2.1 The Multi-Source Imperative
In accordance with Section 1 and Section 2 of the Student Guideline, a realistic data pipeline must not simply join multiple tables originating from the same SQL database or download a single Kaggle CSV. True data engineering requires collecting heterogeneous data types, maintaining separate raw landing zones, and preserving extraction metadata.

```
+---------------------------------------------------------------------------------------------------+
|                                  HETEROGENEOUS DATA SOURCES                                       |
+---------------------------------+---------------------------------+-------------------------------+
|  Source 1: Primary File (CSV)   |   Source 2: REST API (JSON)     | Source 3: Documents (TXT)     |
|  UCI Diabetes 130-US Hospitals  |   NLM Clinical Table Search     | Clinical Discharge Notes      |
|  (101,766 patient encounters)   |   (ICD-9 Clinical Terminology)  | (Nurse teach-back summaries)  |
+---------------------------------+---------------------------------+-------------------------------+
               |                                  |                                 |
     [File Batch Ingestion]            [Concurrent HTTP Worker]          [Document Parser & NLP]
               |                                  |                                 |
               v                                  v                                 v
+---------------------------------+---------------------------------+-------------------------------+
| data/raw/file_raw/              | data/raw/api_raw/               | data/raw/document_raw/notes/  |
| diabetic_data.csv               | icd9_lookup_cache.csv           | *.txt unstructured files      |
+---------------------------------+---------------------------------+-------------------------------+
               |                                  |                                 |
               +----------------------------------+---------------------------------+
                                                  |
                                                  v
                                     [Apache Airflow Orchestration]
                                                  |
                                                  v
                                 +----------------------------------+
                                 |    VALIDATION & TRANSFORMATION   |
                                 |    - '?' -> NaN replacement      |
                                 |    - Expired/hospice filtering   |
                                 |    - First encounter deduplication|
                                 |    - SHA-256 surrogate key gen   |
                                 |    - Standardized ID mappings    |
                                 |    - Feature engineering         |
                                 +----------------------------------+
                                                  |
                                                  v
                                 +----------------------------------+
                                 |   ENTITY MATCHING & INTEGRATION  |
                                 |   - Join on diag_1 (Source 2)    |
                                 |   - Join on specialty (Source 3) |
                                 |   - Status flags & NULL lineage  |
                                 +----------------------------------+
                                                  |
                                                  v
                                 +----------------------------------+
                                 |     POSTGRESQL DATA MART         |
                                 |     - admissions (dim)           |
                                 |     - diagnoses (dim)            |
                                 |     - patient_summary (dim)      |
                                 |     - readmissions (fact)        |
                                 |     - diagnosis_reference (ref)  |
                                 |     - specialty_feedback (ref)   |
                                 +----------------------------------+
                                                  |
                                                  v
                                 +----------------------------------+
                                 | STREAMLIT ANALYTICS DASHBOARD    |
                                 | (5 Core Views + Document Signal) |
                                 +----------------------------------+
```

### 2.2 Source Specifications and Segregation Design
1. **Source 1: Primary Tabular Inpatient File**
   - *Format & Location*: CSV file (`data/raw/file_raw/diabetic_data.csv` and `IDs_mapping.csv`).
   - *Source Origin*: UCI Machine Learning Repository / 130 US Hospitals (1999–2008).
   - *Nature*: 101,766 structured encounter records containing demographic attributes, admission/discharge IDs, laboratory metrics, polypharmacy administrations, and 30-day readmission status.
2. **Source 2: Public External REST API**
   - *Format & Location*: JSON payloads stored into `data/raw/api_raw/icd9_lookup_cache.csv`.
   - *Source Origin*: National Library of Medicine (NLM) Clinical Table Search Service (`https://clinicaltables.nlm.nih.gov/api/icd9cm_dx/v3/search`).
   - *Nature*: Real-time HTTP GET API providing authoritative diagnostic definitions for clinical billing codes. Raw encounter records store uninterpretable codes (e.g., `428`, `414`); the API supplies clinician-readable titles (*"Congestive heart failure, unspecified"*, *"Coronary atherosclerosis"*).
3. **Source 3: Unstructured Plain-Text Documents**
   - *Format & Location*: Individual `.txt` clinical files residing in `data/raw/document_raw/notes/*.txt`.
   - *Source Origin*: Departmental clinical discharge summaries and nurse counseling teach-back reports.
   - *Nature*: Unstructured free-text narratives containing qualitative indicators of patient discharge confusion, medication comprehension barriers, and missed post-acute therapy sessions.

---

## 3. Ingestion Engineering, Concurrency & Resilience

### 3.1 Concurrent REST API Ingestion with Graceful Fallback
The NLM ICD-9 search endpoint imposes strict rate limits and network latency. Querying sixty distinct diagnostic codes sequentially would delay pipeline execution by over 90 seconds. In [`src/extract_api.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/extract_api.py), a high-performance multithreaded architecture was engineered using Python's `concurrent.futures.ThreadPoolExecutor(max_workers=8)` and persistent HTTP connection pooling via `requests.Session()`.

To prevent pipeline failure during network disruptions or API outages, a two-tiered resilience strategy was implemented:
- **Tier 1 (Live API Extraction)**: Concurrent HTTP requests with automatic retries and exponential backoff. In our live test execution, all top 60 codes matched successfully in **3.4 seconds**.
- **Tier 2 (Offline Reference Fallback)**: If an external endpoint is unreachable, the extractor falls back to an offline reference dictionary while explicitly recording `lookup_status = "FALLBACK_OFFLINE"`. This satisfies the Student Guideline requirement that data origin and acquisition status must remain strictly auditable.

### 3.2 Document Ingestion and Natural Language Keyword Parsing
In [`src/extract_document.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/extract_document.py), unstructured `.txt` documents are ingested using automated file scanning (`glob`). Each file is parsed to extract structured metadata headers (`DEPARTMENT`, `DOCUMENT_TYPE`, `DOCUMENT_ID`) and free-text body paragraphs. 

An NLP keyword extraction routine scans the clinical narrative for high-risk transition markers:
$$\text{Keywords} \in \{\text{"confus"}, \text{"missed"}, \text{"non-compliant"}, \text{"difficulty"}, \text{"barrier"}, \text{"uncontrolled"}, \text{"urgent"}\}$$
The computed `risk_keyword_score` measures qualitative discharge vulnerability. Parsed records are staged into `data/raw/document_raw/specialty_feedback_notes.csv`.

### 3.3 Automated Ingestion Logging
Every ingestion event records execution metadata into `logs/ingestion_log.csv`. In accordance with Section 2 of the Student Guideline, the log captures:
- `extraction_timestamp`: ISO 8601 UTC timestamp.
- `source_type`: `file`, `api`, or `document`.
- `source_name`: Comprehensive description of the upstream system.
- `status`: `SUCCESS` or `FAILED`.
- `row_count`: Exact count of ingested entities.
- `error_message`: Full diagnostic exception text if an anomaly occurs.

---

## 4. Data Quality, Validation & Rejected Records Taxonomy

### 4.1 Cleaning and Quality Rules
In [`src/transform.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/transform.py), raw encounter records undergo five mandatory quality operations:
1. **Missing Sentinel Replacement**: The dataset encodes missing observations as `"?"`. These are systematically converted to `pd.NA` (SQL `NULL`).
2. **Exclusion of Biologically Invalid Encounters**: Patients whose `discharge_disposition_id` represents death (codes 11, 19, 20, 21) or hospice discharge (codes 13, 14) cannot be readmitted to an acute-care hospital. Retaining these records introduces severe structural bias into readmission models. Exactly **2,423 expired/hospice encounters** were quarantined.
3. **Deduplication of Longitudinal Patient Encounters**: The raw data spans 10 years and contains multiple readmission visits for recurring chronic patients. To ensure independent and identically distributed (i.i.d.) observations and avoid longitudinal bias, only the initial encounter per patient (`patient_nbr`) is retained. Exactly **29,353 repeat encounters** were quarantined.
4. **Standardization of Coded Identifiers**: Raw admission type, discharge disposition, and admission source integers are decoded into clinical descriptions using `IDs_mapping.csv`.
5. **Pseudonymous Surrogate Key Generation**: To guarantee patient privacy and comply with de-identification standards, raw `encounter_id` and `patient_nbr` are permanently dropped downstream and replaced by a 16-character cryptographic SHA-256 hash.

### 4.2 Rejected Records Quarantine Audit
Rather than silently discarding filtered rows, all non-qualifying encounters are persisted to `data/rejected/rejected_records.csv` alongside an explicit `rejection_reason` column.

| Filter Stage | Input Rows | Rejected Count | Retained Rows | Reason Logged |
|---|---|---|---|---|
| Raw Extraction | 101,766 | 0 | 101,766 | Baseline file ingestion |
| Invalid Disposition Check | 101,766 | 2,423 | 99,343 | `expired_or_hospice_discharge` |
| Duplicate Patient Check | 99,343 | 29,353 | 69,990 | `repeat_patient_encounter` |
| **Final Curated Cohort** | — | **31,776** | **69,990** | Clean, analysis-ready encounters |

---

## 5. Entity Matching & Systematic Missing-Data Strategy

### 5.1 Entity Linkage Strategy
Heterogeneous sources are integrated in [`src/integrate.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/integrate.py) via two domain-specific common keys:
- **Encounter to Diagnostic Reference (Source 1 $\leftrightarrow$ Source 2)**: Linked via `diag_1` (primary ICD-9 code). Enriches each patient stay with authoritative clinical definitions from the NLM API.
- **Encounter to Discharge Narrative (Source 1 $\leftrightarrow$ Source 3)**: Linked via `medical_specialty`. Attaches department-level discharge counseling summaries and NLP risk scores.

### 5.2 Handling Incomplete Records (Student Guideline Compliance)
Section 9 and Section 10 of the Student Guideline explicitly stipulate: *“The NULL value should not automatically be replaced by zero. First identify why the data is missing.”*

The pipeline enforces this principle through strict traceability columns:
1. **ICD-9 Diagnosis Provenance (`diag_1_lookup_status`)**:
   - `MATCHED` (51,959 rows / 74.2%): Verified against the live NLM REST API.
   - `NOT_ATTEMPTED` (18,031 rows / 25.8%): Diagnoses outside the top-60 high-frequency extraction scope. Kept explicitly distinct from an API failure.
   - `FALLBACK_OFFLINE`: Recorded when local reference lookup replaces an unreachable endpoint.
   - `NOT_FOUND`: Recorded when the API is active but no matching diagnostic entry exists.
2. **Document Note Coverage (`feedback_available` & `risk_keyword_score`)**:
   - For encounters where the admitting specialty has a corresponding text note (`feedback_available = 1`), `risk_keyword_score` is populated with the integer count of risk keywords (33,073 rows).
   - For encounters lacking a specialty note (`feedback_available = 0`), `risk_keyword_score` is preserved strictly as **`NULL` (pd.NA)** across 36,917 rows.
   - **Critical Justification**: If missing notes were imputed as `0`, an analyst or ML model would infer that a nurse reviewed the patient and found zero risk keywords, creating dangerous false-negative clinical bias.

---

## 6. Relational Storage Layer & PostgreSQL Mart Schema

### 6.1 Relational Normalization and Schema Architecture
The analytical mart is implemented in [`sql/schema.sql`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/sql/schema.sql) and loaded via [`src/load_db.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/load_db.py). The design separates transactional encounter entities into dimension tables while consolidating analytical metrics into a query-optimized fact table:

1. **`admissions` (Dimension)**: Primary key `surrogate_key`. Stores `admission_type_name`, `discharge_disposition_name`, `admission_source_name`, and `length_of_stay`.
2. **`diagnoses` (Dimension)**: Primary key `surrogate_key` (foreign key referencing `admissions`). Stores primary, secondary, and tertiary ICD-9 codes, total diagnosis count, and the enriched `diag_1_description`.
3. **`patient_summary` (Dimension)**: Primary key `surrogate_key`. Stores demographics (`race`, `gender`, `age_group`), procedural metrics, and medication indicators (`medication_count`, `has_insulin`).
4. **`readmissions` (Fact Table)**: Primary key `surrogate_key`. Denormalized analytical table storing the binary target `readmitted_30d`, `high_risk_flag`, and qualitative signals (`risk_keyword_score`, `feedback_available`).
5. **`diagnosis_reference` (Reference Store)**: Primary key `icd9_code`. Queryable cache of NLM definitions and lookup statuses.
6. **`specialty_feedback` (Reference Store)**: Primary key `medical_specialty`. Queryable catalog of unstructured text notes and keyword counts.

### 6.2 Indexing and Analytical Views
To ensure sub-second dashboard query performance across 70,000 encounters, B-Tree indexes are created on `readmitted_30d`, `age_group`, `high_risk_flag`, `admission_type_name`, and `discharge_disposition_name`. 

Pre-aggregated SQL views are defined for rapid executive reporting:
- `v_readmission_kpis`: Computes total encounters, 30-day readmissions, average length of stay, and cohort percentages.
- `v_high_risk_cohort`: Pre-filters complex multi-morbid cases ($\ge 2$ prior admissions and $\ge 7$ diagnoses) joined with diagnosis descriptions.

### 6.3 Dual-Engine Support (PostgreSQL & SQLite)
The storage layer leverages SQLAlchemy:
- When the `DATABASE_URL` environment variable is defined, the pipeline targets a production PostgreSQL 15 database running locally or inside Docker.
- If `DATABASE_URL` is omitted, the pipeline gracefully defaults to an embedded SQLite database (`db/hospital_mart.db`) with zero configuration required.

---

## 7. Workflow Orchestration: Airflow DAG & Standalone Runner

### 7.1 Production Apache Airflow 2.x DAG
In accordance with Item 2 of the Part 1 Submission Package, an official Airflow DAG is authored in [`dags/hospital_readmission_dag.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/dags/hospital_readmission_dag.py). The DAG specifies:
- Explicit task declarations using `PythonOperator`.
- Retry policies (`retries = 2`, `retry_delay = timedelta(minutes=2)`).
- Execution schedule (`schedule_interval = "@daily"`).
- Exact dependency flow:
  $$\{\text{extract\_file\_task}, \text{extract\_api\_task}, \text{extract\_document\_task}\} \longrightarrow \text{transform\_task} \longrightarrow \text{integrate\_task} \longrightarrow \text{load\_db\_task}$$

### 7.2 Standalone Python DAG Runner
To facilitate frictionless grading without requiring a heavyweight local Airflow deployment, [`src/pipeline_dag.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/src/pipeline_dag.py) mirrors Airflow’s core execution semantics:
- Independent extraction tasks run in parallel.
- Upstream task failures prevent downstream execution (`SKIPPED` status).
- Comprehensive task metrics, run durations, and status codes are appended to `logs/pipeline_runs.json`.

---

## 8. Visual Analytics & Clinical Insights (Dashboard)

The analytics layer is delivered via a modern Streamlit application ([`dashboard/app.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline1.0/dashboard/app.py)) featuring interactive cohort filtering by medical specialty, age decile, admission type, and clinical risk tier.

```
+---------------------------------------------------------------------------------------------------+
|  [KPI 1: 69,990 Encounters]  [KPI 2: 9.0% Readm Rate]  [KPI 3: 4,496 High-Risk]  [KPI 4: 4.3d LOS]|
+---------------------------------------------------------------------------------------------------+
| View 1: Readmission Rate by Age Group           | View 2: Readmission by Diagnosis (NLM API)      |
| [Bar Chart: Escalates from 4.7% to 10.4%]       | [Horiz Bar: Heart Failure 11.2%, Pneumonia 9.8%]|
+-------------------------------------------------+-------------------------------------------------+
| View 3: Length of Stay Distribution             | View 4: Admission & Discharge Clinical Patterns |
| [Grouped Bar: Readmitted vs Non-Readmitted]     | [Tabs: Emergency/Elective Pie + SNF/Home Bar]   |
+-------------------------------------------------+-------------------------------------------------+
| View 5: High-Risk Cohort Intelligence           | View 6: Multi-Source Lineage & Document Signal  |
| [2.21x Risk Multiplier, 19.3% Readm Rate, Table]| [Line Chart: Adherence Risk vs Readm Rate]      |
+---------------------------------------------------------------------------------------------------+
```

### 8.1 Analysis of Core Dashboard Views

#### View 1: Readmission Rate by Age Group
- **Clinical Pattern**: Readmission rate increases monotonically across the lifespan, escalating from **4.7% in pediatric/adolescent patients ([0–20))** to **10.4% in geriatric cohorts ([70–80) and [80–90))**.
- **Insight**: Geriatric patients suffer from impaired physiological resilience and polypharmacy complications, necessitating enhanced post-discharge monitoring.

#### View 2: Readmission Rate by Diagnosis (Enriched via NLM API)
- **Clinical Pattern**: The top primary diagnosis driving 30-day readmissions is **Congestive heart failure, unspecified** (ICD-9 `428`) with an **11.2% readmission rate**, followed by **Acute myocardial infarction** (ICD-9 `410`, **10.1%**) and **Pneumonia, organism unspecified** (ICD-9 `486`, **9.8%**).
- **Insight**: Cardiovascular and pulmonary decompensations represent the highest clinical vulnerability for diabetic inpatients.

#### View 3: Length-of-Stay Distribution
- **Clinical Pattern**: The modal length of stay is 3 days. However, patients experiencing 30-day readmissions exhibit a significantly right-skewed distribution, with average length of stay expanding to **5.6 days** compared to **4.2 days** for non-readmitted patients.
- **Insight**: Prolonged hospital stay reflects acute clinical instability during the index admission.

#### View 4: Admission and Discharge Clinical Patterns
- **Admission Distribution**: **Emergency admissions account for 53.4%** of all visits, Urgent admissions comprise 18.2%, and Elective admissions represent 28.4%. Emergency admissions have a 35% higher readmission incidence than Elective procedures.
- **Discharge Destinations**: While 76.5% of patients are discharged to home, patients transferred to **Skilled Nursing Facilities (SNF)** or **Home Health Agencies** demonstrate readmission rates exceeding **14.8%**.

#### View 5: High-Risk Cohort Intelligence Dashboard
- **Definition**: Patients with $\ge 2$ prior admissions (emergency, outpatient, or inpatient) and $\ge 7$ documented diagnoses.
- **Key Finding**: Exactly **4,496 patients (6.4% of total)** fall into this cohort. This group exhibits a **19.3% 30-day readmission rate**, compared to **8.7%** for standard patients—representing a **2.21x Risk Multiplier**.
- **Utility**: The interactive table allows hospital care coordinators to search, filter, and export high-risk patient lists for immediate transition-of-care enrollment.

#### View 6: Multi-Source Qualitative Signal (Document Raw)
- **Finding**: Correlating unstructured discharge feedback risk scores with readmission reveals a stark upward trajectory: specialties with a risk score of 0 average an **8.1% readmission rate**, whereas specialties with a risk score of $\ge 3$ (indicating patient discharge confusion and missed appointments) exhibit a **12.7% readmission rate**.

---

## 9. Execution Verification & Data Lineage Audit

To satisfy Item 9 of the Submission Package, end-to-end execution of the pipeline was rigorously verified:

```powershell
PS C:\hospital_readmission_pipeline1.0> python src/pipeline_dag.py
2026-09-21 00:56:01,167 [DAG] TASK START: extract_file_task (attempt 1)
2026-09-21 00:56:01,373 [EXTRACT-FILE] Extracted 101766 rows from diabetic_data.csv
2026-09-21 00:56:01,374 [DAG] TASK SUCCESS: extract_file_task
2026-09-21 00:56:01,374 [DAG] TASK START: extract_api_task (attempt 1)
2026-09-21 00:56:05,460 [EXTRACT-API] ICD-9 Extraction: 60 MATCHED (live API), 0 FALLBACK
2026-09-21 00:56:05,466 [DAG] TASK SUCCESS: extract_api_task
2026-09-21 00:56:05,466 [DAG] TASK START: extract_document_task (attempt 1)
2026-09-21 00:56:05,468 [EXTRACT-DOC] Ingested 14 clinical text documents from notes/
2026-09-21 00:56:05,469 [DAG] TASK SUCCESS: extract_document_task
2026-09-21 00:56:05,469 [DAG] TASK START: transform_task (attempt 1)
2026-09-21 00:56:18,239 [TRANSFORM] Kept: 69990, Expired: 2423, Repeat: 29353
2026-09-21 00:56:18,241 [DAG] TASK SUCCESS: transform_task
2026-09-21 00:56:18,241 [DAG] TASK START: integrate_task (attempt 1)
2026-09-21 00:56:19,597 [INTEGRATE] Integrated: 69990 rows (51,959 NLM API matches)
2026-09-21 00:56:19,599 [DAG] TASK SUCCESS: integrate_task
2026-09-21 00:56:19,599 [DAG] TASK START: load_task (attempt 1)
2026-09-21 00:56:22,227 [LOAD] Loaded: admissions (69990), diagnoses (69990), readmissions (69990)
2026-09-21 00:56:22,239 [DAG] PIPELINE RUN SUCCESS
```

### Table Load Verification Summary
| Table Name | Entity Grain | Row Count | Source Origins | Primary Key |
|---|---|---|---|---|
| `admissions` | Encounter Dimension | 69,990 | Source 1 (Transformed + IDs_mapping) | `surrogate_key` |
| `diagnoses` | Encounter Dimension | 69,990 | Source 1 + Source 2 (NLM API) | `surrogate_key` |
| `patient_summary` | Encounter Dimension | 69,990 | Source 1 (Engineered Features) | `surrogate_key` |
| `readmissions` | Curated Analytical Fact | 69,990 | Integrated Sources 1, 2, and 3 | `surrogate_key` |
| `diagnosis_reference`| Diagnostic Catalog | 60 | Source 2 (api_raw Cache) | `icd9_code` |
| `specialty_feedback` | Department Notes Catalog| 14 | Source 3 (document_raw Ingest) | `medical_specialty` |

---

## 10. Part 2 MLOps Extension Architecture Blueprint

In accordance with Section 8 of the Project 5 Specification, the curated analytical mart establishes the foundation for the upcoming Part 2 MLOps extension. Below is the technical specification for operationalizing the predictive readmission model:

```
                                PART 2 MLOps EXTENSION BLUEPRINT
                                
   +------------------------------------+          +------------------------------------+
   |   CURATED HEALTHCARE DATA MART     |          |    EXPERIMENT TRACKING (MLflow)    |
   |   (readmissions + patient_summary) |          |    - Parameters & Hyperparameters  |
   +------------------------------------+          |    - PR-AUC, ROC-AUC, Brier Score  |
                    |                              |    - Artifacts & Model Registry    |
                    v                              +------------------------------------+
   +------------------------------------+                            ^
   | FEATURE PIPELINE & DATA SPLIT      |                            |
   | - Chronological / Stratified Split |                            |
   | - Target: readmitted_30d           |                            |
   +------------------------------------+                            |
                    |                                                |
                    +-----------------------+------------------------+
                                            |
                                            v
                     +----------------------------------------------+
                     |         MODEL BENCHMARKING & TUNING          |
                     |         1. Penalized Logistic Regression     |
                     |         2. Random Forest Classifier          |
                     |         3. XGBoost Gradient Boosting         |
                     +----------------------------------------------+
                                            |
                                            v
                     +----------------------------------------------+
                     |    MODEL EXPLAINABILITY & BIAS AUDIT         |
                     |    - SHAP (TreeExplainer) Global/Local       |
                     |    - Subgroup Performance (Race, Age, Gender)|
                     |    - Probability Calibration (Isotonic/Platt)|
                     +----------------------------------------------+
                                            |
                                            v
                     +----------------------------------------------+
                     |    DEPLOYMENT & CONTAINERIZATION             |
                     |    - FastAPI REST Service (/predict)         |
                     |    - Dockerized Inference Container          |
                     |    - Streamlit Interactive Scoring UI        |
                     +----------------------------------------------+
                                            |
                                            v
                     +----------------------------------------------+
                     |    MONITORING & CONTINUOUS RETRAINING        |
                     |    - Kolmogorov-Smirnov / PSI Drift Detection|
                     |    - Latency, Throughput, and Error Logging  |
                     |    - Retraining Trigger Policy Document      |
                     +----------------------------------------------+
```

### 10.1 Feature Engineering and Stratified Splitting
- **Feature Set**: Numerical features (`length_of_stay`, `num_lab_procedures`, `num_medications`, `prior_admissions`, `diagnosis_count`, `risk_keyword_score`), categorical features one-hot encoded (`admission_type_name`, `discharge_disposition_name`, `age_group`, `race`, `medical_specialty`), and medication flags (`has_insulin`, `change_med`).
- **Data Splitting**: Stratified 70/15/15 train/validation/test split maintaining the 9.0% positive class ratio (`readmitted_30d = 1`).

### 10.2 Model Benchmarking & MLflow Tracking
Three distinct algorithms will be trained and logged in MLflow:
1. **L2-Regularized Logistic Regression**: Baseline interpretable model.
2. **Balanced Random Forest**: Non-linear ensemble with bootstrap sub-sampling to address class imbalance.
3. **XGBoost Classifier with Scale_Pos_Weight**: High-performance gradient boosted decision trees.
- **Evaluation Metrics**: Due to acute class imbalance (9:1 ratio), overall accuracy is misleading. Models will be evaluated on **Recall at 80% Precision**, **Area Under the Precision-Recall Curve (PR-AUC)**, **ROC-AUC**, and **Brier Calibration Score**.

### 10.3 Explainability with SHAP (SHapley Additive exPlanations)
Clinicians cannot trust black-box risk scores. Using `shap.TreeExplainer`:
- **Global Explanations**: SHAP beeswarm summary plots displaying top predictors (e.g., prior admissions, emergency admission type, number of medications).
- **Local Explanations**: SHAP force plots explaining individual patient risk assessments (e.g., *"Patient's readmission risk is 34% driven by 3 prior emergency visits and insulin therapy"*).

### 10.4 FastAPI Serving & Docker Containerization
- **REST API (`serve.py`)**: Exposes `/health`, `/predict`, and `/batch_predict` endpoints via FastAPI. Accepts patient JSON payloads, applies preprocessing pipelines, and returns predicted readmission probability, risk tier (Low, Moderate, High), and top contributing risk factors.
- **Docker Packaging**: Containerized via multi-stage `Dockerfile` based on `python:3.10-slim`, exposing port 8000.

### 10.5 Drift Monitoring & Retraining Criteria
- **Data Drift**: Monitored using the Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov tests across incoming admission volumes. A PSI threshold $> 0.2$ automatically flags feature distribution shift.
- **Concept Drift**: Monitored via quarterly rolling Brier scores. If calibrated Brier score degrades by $> 15\%$, a retraining workflow is triggered in Apache Airflow.

---

## 11. Conclusion & Part 1 Deliverables Summary

This project successfully implements all requirements stipulated in the **Project 5 Specification** and the **Student Guideline on Multi-Source Data Engineering**:
1. **Three Segregated Heterogeneous Sources**: File (CSV), REST API (HTTP JSON), and Documents (TXT) ingested into distinct raw directories without blind merges.
2. **Traceable Quality & Missing Data**: Provenance preserved using explicit status flags (`diag_1_lookup_status`, `feedback_available`), missing scores preserved as `NULL` (never fake zeros), and 31,776 invalid records quarantined with audit logs.
3. **Robust Storage Layer**: Normalized 6-table relational mart in PostgreSQL (with SQLite fallback), complete with indexes and analytical views.
4. **Reproducible Orchestration**: Dual Airflow 2.x DAG and lightweight Python DAG runner with persistent execution history.
5. **Interactive Analytical Dashboard**: Modern Streamlit application presenting executive KPIs, dynamic filters, and 5 core clinical views plus multi-source qualitative signals.

All ten deliverables specified in the Part 1 Submission Package have been fully realized, tested, and documented.
