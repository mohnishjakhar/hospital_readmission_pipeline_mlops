# Data Dictionary — Hospital Readmission Analytical Mart (Multi-Source)

## 1. Multi-Source Architecture and Raw Segregation

In accordance with the project guidelines, data is collected from **three genuinely different source types**, ingested via source-specific mechanisms, and maintained in isolated, segregated raw zones before any transformations or joins are performed.

| # | Source Name | Source Type | Raw Format | Ingestion Method | Raw Destination Landing Zone | Source Grain |
|---|---|---|---|---|---|---|
| **1** | UCI Diabetes 130-US Hospitals (1999–2008) | Structured File | CSV (101,766 rows) | Batch CSV Ingestion (`pandas`) | `data/raw/file_raw/diabetic_data.csv` | 1 record per patient encounter |
| **2** | NLM Clinical Table Search Service (`clinicaltables.nlm.nih.gov`) | Public REST API | JSON via HTTP GET | Concurrent API Connector (`requests` + `ThreadPoolExecutor`) | `data/raw/api_raw/icd9_lookup_cache.csv` | 1 record per unique ICD-9 diagnosis code |
| **3** | Departmental Clinical Discharge & Feedback Notes | Unstructured Documents | Plain Text (.txt files) | Document Ingestion & NLP Parser (`glob` + `re`) | `data/raw/document_raw/notes/*.txt` | 1 document per admitting medical specialty |

### Rationale and Clinical Value of Each Source
- **Source 1 (File - Encounter Records)**: Serves as the core clinical spine. Captures patient demographics, admission/discharge IDs, lab tests, medications, and the primary 30-day readmission outcome label.
- **Source 2 (API - NLM Clinical Definitions)**: Resolves opaque, numeric ICD-9 codes (e.g., `428`, `414`, `250.8`) into official clinical terminology (e.g., *"Congestive heart failure, unspecified"*, *"Coronary atherosclerosis"*). Without this API, clinical users cannot interpret diagnostic patterns.
- **Source 3 (Documents - Qualitative Discharge Summaries)**: Ingests unstructured clinical text documents capturing nurse teach-back notes, discharge instruction comprehension, and patient compliance barriers. Extracts qualitative risk signals that structured billing codes fail to capture.

---

## 2. Standardized Mappings & Reference Lookups

In raw Source 1, admission, discharge, and source fields are recorded as numeric integers. These are decoded into standardized clinical descriptions via `data/raw/file_raw/IDs_mapping.csv`:

### Admission Type Mapping (`admission_type_name`)
- `1`: Emergency
- `2`: Urgent
- `3`: Elective
- `4`: Newborn
- `5`: Not Available
- `7`: Trauma Center
- `6, 8`: NULL / Not Mapped / Unknown

### Discharge Disposition Mapping (`discharge_disposition_name`)
- `1`: Discharged to home
- `2`: Discharged/transferred to another short term general hospital
- `3`: Discharged/transferred to SNF (Skilled Nursing Facility)
- `4`: Discharged/transferred to ICF (Intermediate Care Facility)
- `5`: Discharged/transferred to another inpatient care institution
- `6`: Discharged/transferred to home with home health service
- `7`: Left AMA (Against Medical Advice)
- `11, 13, 14, 19, 20, 21`: Expired / Hospice (Excluded from analytical cohort)
- `22`: Discharged/transferred to rehab facility
- `23`: Discharged/transferred to long term care hospital

---

## 3. Entity Matching and Integration Keys

| Join Relationship | Join Key | Cardinality | Purpose |
|---|---|---|---|
| Source 1 ↔ Source 2 | `diag_1` (ICD-9 code) | N Encounters → 1 Code Description | Enrich encounter diagnosis with official NLM clinical name |
| Source 1 ↔ Source 3 | `medical_specialty` | N Encounters → 1 Specialty Document | Attach qualitative discharge risk score and note excerpt |

### Pseudonymous Surrogate Key
To protect patient privacy while maintaining referential integrity across normalized relational tables, direct hospital identifiers (`encounter_id` and `patient_nbr`) are hashed into a 16-character hexadecimal surrogate key:
$$\text{surrogate\_key} = \text{SHA256}(\text{encounter\_id} \mathbin{\Vert} \text{"-"} \mathbin{\Vert} \text{patient\_nbr})[:16]$$

---

## 4. Missing-Data Lineage and Quality Rules

Per Section 9 of the Student Guideline, missing data across heterogeneous sources must **never be blindly replaced with zero**. The pipeline preserves source lineage through explicit status indicators:

| Feature | Traceability Column | Allowed Values | Handling Strategy & Clinical Justification |
|---|---|---|---|
| `diag_1_description` | `diag_1_lookup_status` | `MATCHED`<br/>`FALLBACK_OFFLINE`<br/>`NOT_FOUND`<br/>`NOT_ATTEMPTED` | If code was outside the top-N extraction scope, status is `NOT_ATTEMPTED` rather than an erroneous API failure. If API was unreachable, `FALLBACK_OFFLINE` documents the local reference lookup. |
| `feedback_text` | `feedback_available` | `1` (Present)<br/>`0` (Missing) | Binary flag indicating whether the admitting specialty has a documented discharge summary. |
| `risk_keyword_score` | `feedback_available` | Integer $\ge 0$ if present, else `NULL` | If `feedback_available=0`, the risk score is kept strictly as `NULL` (pd.NA). Setting this to 0 would falsely indicate that a note was reviewed and contained zero risk keywords. |

---

## 5. Storage Layer: Relational Mart Schema

The storage layer is implemented in PostgreSQL (with SQLite fallback) across six structured tables:

### 5.1 Table: `admissions` (Dimension)
| Column Name | Data Type | Constraint | Description |
|---|---|---|---|
| `surrogate_key` | VARCHAR(64) | PRIMARY KEY | Pseudonymous encounter identifier |
| `admission_type_id` | INTEGER | | Coded admission category |
| `admission_type_name` | VARCHAR(128) | | Standardized label (e.g., Emergency, Elective) |
| `admission_source_id` | INTEGER | | Coded referral source |
| `admission_source_name`| VARCHAR(128) | | Standardized source (e.g., Emergency Room, Clinic Referral) |
| `discharge_disposition_id` | INTEGER | | Coded discharge disposition |
| `discharge_disposition_name` | VARCHAR(256) | | Standardized outcome (e.g., Discharged to home, SNF) |
| `length_of_stay` | INTEGER | NOT NULL | Total inpatient days (1–14) |
| `medical_specialty` | VARCHAR(128) | | Admitting medical department |
| `payer_code` | VARCHAR(32) | | Coded health insurer/payer |

### 5.2 Table: `diagnoses` (Dimension, Enriched by Source 2)
| Column Name | Data Type | Constraint | Description |
|---|---|---|---|
| `surrogate_key` | VARCHAR(64) | PRIMARY KEY, FK | References `admissions(surrogate_key)` |
| `diag_1` | VARCHAR(16) | | Primary ICD-9 diagnosis code |
| `diag_2` | VARCHAR(16) | | Secondary ICD-9 diagnosis code |
| `diag_3` | VARCHAR(16) | | Tertiary ICD-9 diagnosis code |
| `diagnosis_count` | INTEGER | | Total diagnoses recorded for encounter |
| `diag_1_description` | TEXT | | Clinical terminology from NLM REST API |
| `diag_1_lookup_status` | VARCHAR(32) | | Ingestion status (`MATCHED`, `NOT_ATTEMPTED`) |

### 5.3 Table: `patient_summary` (Dimension)
| Column Name | Data Type | Constraint | Description |
|---|---|---|---|
| `surrogate_key` | VARCHAR(64) | PRIMARY KEY, FK | References `admissions(surrogate_key)` |
| `race` | VARCHAR(64) | | Patient racial category (or Unknown) |
| `gender` | VARCHAR(32) | | Patient gender (Male, Female, Unknown) |
| `age_group` | VARCHAR(32) | | 10-year age bracket (e.g., `[70-80)`) |
| `num_lab_procedures` | INTEGER | | Number of diagnostic lab procedures performed |
| `num_procedures` | INTEGER | | Number of non-lab clinical procedures |
| `medication_count` | INTEGER | | Engineered: Count of active diabetes medications prescribed |
| `prior_admissions` | INTEGER | | Engineered: Sum of prior outpatient, ER, and inpatient visits |
| `high_risk_flag` | INTEGER | | Engineered: 1 if prior admissions $\ge 2$ and diagnoses $\ge 7$ |
| `has_insulin` | INTEGER | | 1 if insulin was prescribed during stay |
| `change_med` | INTEGER | | 1 if diabetic medications were altered during stay |
| `has_diabetes_med` | INTEGER | | 1 if any diabetes medication was prescribed |

### 5.4 Table: `readmissions` (Fact Table, Enriched by Source 3)
| Column Name | Data Type | Constraint | Description |
|---|---|---|---|
| `surrogate_key` | VARCHAR(64) | PRIMARY KEY, FK | References `admissions(surrogate_key)` |
| `readmitted` | VARCHAR(16) | NOT NULL | Original dataset outcome (`NO`, `<30`, `>30`) |
| `readmitted_30d` | INTEGER | NOT NULL | Binary prediction target: 1 if readmitted within 30 days |
| `age_group` | VARCHAR(32) | | Denormalized age category for indexing |
| `medical_specialty` | VARCHAR(128) | | Admitting medical department |
| `admission_type_name` | VARCHAR(128) | | Standardized admission label |
| `discharge_disposition_name` | VARCHAR(256) | | Standardized discharge destination |
| `length_of_stay` | INTEGER | | Total inpatient days |
| `diagnosis_count` | INTEGER | | Total diagnoses count |
| `prior_admissions` | INTEGER | | Total prior acute care encounters |
| `high_risk_flag` | INTEGER | | High-risk cohort classification indicator |
| `risk_keyword_score` | INTEGER | NULLABLE | Source 3: NLP keyword count from discharge note (NULL if no note) |
| `feedback_available` | INTEGER | NOT NULL | Source 3: 1 if note linked, 0 if missing |

### 5.5 Table: `diagnosis_reference` (Reference Store)
| Column Name | Data Type | Description |
|---|---|---|
| `icd9_code` | VARCHAR(16) PRIMARY KEY | Unique ICD-9-CM code queried against NLM API |
| `description` | TEXT | Official NLM Clinical Tables definition |
| `lookup_status` | VARCHAR(32) | Status code (`MATCHED`, `FALLBACK_OFFLINE`, `NOT_FOUND`) |

### 5.6 Table: `specialty_feedback` (Reference Store)
| Column Name | Data Type | Description |
|---|---|---|
| `medical_specialty` | VARCHAR(128) PRIMARY KEY | Admitting department |
| `document_id` | VARCHAR(64) | Source document identifier (e.g., `DOC-CARD-001`) |
| `document_type` | VARCHAR(128) | Type of clinical record (e.g., Discharge Summary) |
| `feedback_text` | TEXT | Ingested plain-text qualitative clinical note |
| `risk_keyword_score` | INTEGER | NLP adherence/confusion risk score |
| `notes_available` | INTEGER | Always 1 for present notes |

---

## 6. Pre-Integration Validation & Quality Rules

1. **Missing Indicator Replacement**: All raw string sentinel values (`"?"`) replaced with standard `NULL`/`pd.NA`.
2. **Invalid Encounter Exclusion**: Encounters where `discharge_disposition_id` indicates patient death or hospice transfer (codes 11, 13, 14, 19, 20, 21) are removed because 30-day hospital readmission cannot biologically occur.
3. **Repeated Encounter Deduplication**: For patients with multiple visits across the 10-year study window, only the initial encounter is retained to prevent repeat-patient longitudinal bias.
4. **Audit Logging**: All excluded encounters are preserved in `data/rejected/rejected_records.csv` alongside the specific exclusion reason (`expired_or_hospice_discharge` or `repeat_patient_encounter`).
