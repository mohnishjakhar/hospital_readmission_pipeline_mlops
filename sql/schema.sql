-- =====================================================================
-- Hospital Readmission Analytics Data Mart - Schema DDL
-- Database: PostgreSQL 15+ (Compatible with SQLite)
-- Purpose: Analytical Data Mart integrating 3 Segregated Healthcare Sources
-- =====================================================================

-- Drop existing tables/views if recreating
DROP VIEW IF EXISTS v_high_risk_cohort CASCADE;
DROP VIEW IF EXISTS v_readmission_kpis CASCADE;
DROP TABLE IF EXISTS readmissions CASCADE;
DROP TABLE IF EXISTS admissions CASCADE;
DROP TABLE IF EXISTS diagnoses CASCADE;
DROP TABLE IF EXISTS patient_summary CASCADE;
DROP TABLE IF EXISTS specialty_feedback CASCADE;
DROP TABLE IF EXISTS diagnosis_reference CASCADE;

-- ---------------------------------------------------------------------
-- 1. Reference Table: diagnosis_reference (Source 2: api_raw)
-- ---------------------------------------------------------------------
CREATE TABLE diagnosis_reference (
    icd9_code VARCHAR(16) PRIMARY KEY,
    description TEXT,
    lookup_status VARCHAR(32) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE diagnosis_reference IS 'ICD-9 diagnosis reference definitions ingested from NLM Clinical Tables REST API';

-- ---------------------------------------------------------------------
-- 2. Reference Table: specialty_feedback (Source 3: document_raw)
-- ---------------------------------------------------------------------
CREATE TABLE specialty_feedback (
    medical_specialty VARCHAR(128) PRIMARY KEY,
    document_id VARCHAR(64),
    document_type VARCHAR(128),
    feedback_text TEXT,
    risk_keyword_score INTEGER,
    notes_available INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE specialty_feedback IS 'Clinical discharge and patient care feedback notes ingested from unstructured text documents';

-- ---------------------------------------------------------------------
-- 3. Dimension Table: admissions (Source 1 transformed)
-- ---------------------------------------------------------------------
CREATE TABLE admissions (
    surrogate_key VARCHAR(64) PRIMARY KEY,
    admission_type_id INTEGER,
    admission_type_name VARCHAR(128),
    admission_source_id INTEGER,
    admission_source_name VARCHAR(128),
    discharge_disposition_id INTEGER,
    discharge_disposition_name VARCHAR(256),
    length_of_stay INTEGER NOT NULL,
    medical_specialty VARCHAR(128),
    payer_code VARCHAR(32)
);

CREATE INDEX idx_adm_type ON admissions(admission_type_name);
CREATE INDEX idx_dsch_disp ON admissions(discharge_disposition_name);
CREATE INDEX idx_adm_specialty ON admissions(medical_specialty);

-- ---------------------------------------------------------------------
-- 4. Dimension Table: diagnoses (Enriched with Source 2: api_raw)
-- ---------------------------------------------------------------------
CREATE TABLE diagnoses (
    surrogate_key VARCHAR(64) PRIMARY KEY,
    diag_1 VARCHAR(16),
    diag_2 VARCHAR(16),
    diag_3 VARCHAR(16),
    diagnosis_count INTEGER,
    diag_1_description TEXT,
    diag_1_lookup_status VARCHAR(32),
    CONSTRAINT fk_diag_encounter FOREIGN KEY (surrogate_key) REFERENCES admissions(surrogate_key) ON DELETE CASCADE
);

CREATE INDEX idx_diag1_code ON diagnoses(diag_1);
CREATE INDEX idx_diag1_status ON diagnoses(diag_1_lookup_status);

-- ---------------------------------------------------------------------
-- 5. Dimension Table: patient_summary (Source 1 transformed)
-- ---------------------------------------------------------------------
CREATE TABLE patient_summary (
    surrogate_key VARCHAR(64) PRIMARY KEY,
    race VARCHAR(64),
    gender VARCHAR(32),
    age_group VARCHAR(32),
    num_lab_procedures INTEGER,
    num_procedures INTEGER,
    medication_count INTEGER,
    prior_admissions INTEGER,
    high_risk_flag INTEGER,
    has_insulin INTEGER,
    change_med INTEGER,
    has_diabetes_med INTEGER,
    CONSTRAINT fk_patient_encounter FOREIGN KEY (surrogate_key) REFERENCES admissions(surrogate_key) ON DELETE CASCADE
);

CREATE INDEX idx_patient_age ON patient_summary(age_group);
CREATE INDEX idx_patient_highrisk ON patient_summary(high_risk_flag);

-- ---------------------------------------------------------------------
-- 6. Fact Table: readmissions (Curated Integrated Analytical Mart)
-- ---------------------------------------------------------------------
CREATE TABLE readmissions (
    surrogate_key VARCHAR(64) PRIMARY KEY,
    readmitted VARCHAR(16) NOT NULL,
    readmitted_30d INTEGER NOT NULL,
    age_group VARCHAR(32),
    medical_specialty VARCHAR(128),
    admission_type_name VARCHAR(128),
    discharge_disposition_name VARCHAR(256),
    length_of_stay INTEGER,
    diagnosis_count INTEGER,
    prior_admissions INTEGER,
    high_risk_flag INTEGER,
    risk_keyword_score INTEGER,
    feedback_available INTEGER DEFAULT 0,
    CONSTRAINT fk_readm_encounter FOREIGN KEY (surrogate_key) REFERENCES admissions(surrogate_key) ON DELETE CASCADE
);

CREATE INDEX idx_readm_target ON readmissions(readmitted_30d);
CREATE INDEX idx_readm_specialty ON readmissions(medical_specialty);
CREATE INDEX idx_readm_highrisk ON readmissions(high_risk_flag);
CREATE INDEX idx_readm_los ON readmissions(length_of_stay);

-- ---------------------------------------------------------------------
-- 7. Analytical Views for Reporting & Dashboards
-- ---------------------------------------------------------------------
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
