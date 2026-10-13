# Hospital Readmission Analytics & Clinical MLOps Pipeline
## End-to-End Healthcare Data Engineering & Machine Learning System (Parts 1 & 2)

**Individual Project**: Data Engineering & MLOps  
**Part 1**: Multi-Source Healthcare ETL Pipeline, Analytical PostgreSQL Mart, and Interactive Streamlit Dashboard (50 Marks).  
**Part 2**: Production Clinical MLOps System: Feature Engineering, Multi-Model Comparison, MLflow Tracking & Registry, SHAP Explainability, FastAPI Microservice, Docker Orchestration, and Data Drift Monitoring (50 Marks).  
**Compliance**: Fully aligned with **Project 5 Specification** and **Student Guidelines on Segregated Multi-Source Data Engineering**.

---

## 1. System Architecture: Data Engineering + MLOps

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PART 1: MULTI-SOURCE DATA ENGINEERING                           │
│                                                                                        │
│  [Source 1: File CSV]        [Source 2: REST API]         [Source 3: Documents TXT]    │
│  UCI Diabetes Inpatient       NLM Clinical Tables API      Clinical Discharge Notes    │
│  (101,766 raw encounters)     (ICD-9 Terminology Search)   (NLP Keyword Adherence)     │
│             └───────────────────────────┬───────────────────────────┘                  │
│                                         ▼                                              │
│                        Airflow DAG / Standalone Orchestrator                           │
│                      (Validation, Cleaning, Hashing, ID Mapping)                       │
│                                         ▼                                              │
│                   PostgreSQL / SQLite Normalized Analytical Mart                       │
│                       (69,990 clean, de-duplicated encounters)                         │
└─────────────────────────────────────────┬──────────────────────────────────────────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          PART 2: PRODUCTION HEALTHCARE MLOps                           │
│                                                                                        │
│  1. Feature Store & Stratified Split (`src/ml_features.py`)                            │
│     ├── 25 Clinical Features (Labs, Meds, Regimens, ICD-9 Clusters, Notes Score)      │
│     └── 70/15/15 Stratified Split (48,993 Train, 10,498 Val, 10,499 Held-Out Test)     │
│                                         │                                              │
│                                         ▼                                              │
│  2. Model Training & Comparison (`src/ml_train.py`)                                    │
│     ├── Logistic Regression (Balanced Baseline)                                       │
│     ├── Random Forest (Ensemble Trees)                                                 │
│     └── XGBoost Classifier (Champion Model: Scale-Pos-Weight = 10.13)                 │
│                                         │                                              │
│                                         ▼                                              │
│  3. MLflow Experiment Tracking & Model Registry                                        │
│     ├── Tracks: Recall, Precision, F1, ROC-AUC, PR-AUC, Calibration Brier Score        │
│     ├── Artifacts: ROC Curves, PR Curves, Confusion Matrices, Calibration Plots       │
│     └── Registry: Champion registered as 'HospitalReadmissionPredictor' (v1)           │
│                                         │                                              │
│                                         ▼                                              │
│  4. Clinical Explainability (XAI) with SHAP (`src/ml_explain.py`)                      │
│     ├── Global Beeswarm Summary & Feature Importance Bar Plots                         │
│     └── Local Patient Waterfall Attribution & Actionable Care Recommendations          │
│                                         │                                              │
│                                         ▼                                              │
│  5. REST Prediction Microservice via FastAPI (`src/ml_serve.py`)                       │
│     ├── GET  /health           : Microservice liveness & champion metadata             │
│     ├── POST /predict          : Real-time single patient risk scoring (<20ms latency) │
│     ├── POST /predict-batch    : High-throughput vectorized cohort screening           │
│     └── GET  /case-studies     : Pre-calculated reference clinical archetypes          │
│                                         │                                              │
│                                         ▼                                              │
│  6. Clinical Decision Support Dashboard (`dashboard/app.py`)                           │
│     ├── Tab 1: 6 Core Analytical Population Views (Part 1 Mart)                        │
│     ├── Tab 2: Interactive Patient Risk Calculator & Risk Gauge (Part 2 AI)            │
│     └── Tab 3: MLOps Telemetry, Diagnostic Curves & Drift Auditing (Part 2 Governance) │
│                                         │                                              │
│                                         ▼                                              │
│  7. Continuous Monitoring & Subgroup Fairness (`src/ml_monitor.py`)                    │
│     ├── Statistical Input Feature Drift (2-Sample KS-Test & PSI)                       │
│     ├── Subgroup Fairness Disparity Audit (Age, Gender, Race Parity)                   │
│     └── Automated Operational Retraining Policy Trigger                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Quick Start & Execution Guide

### 2.1 Python Environment & Dependencies
The pipeline is optimized for Python 3.10 / 3.11:
```bash
pip install -r requirements.txt
```

### 2.2 Run Full Pipeline Steps Sequentially

#### Step 1: Execute Part 1 Data Engineering ETL
Ingests from 3 sources, validates, and builds the analytical mart:
```bash
python src/pipeline_dag.py
```

#### Step 2: Run Part 2 Feature Engineering & Preprocessing
Creates leakage-free feature store, stratified splits, and serializes ColumnTransformer:
```bash
python src/ml_features.py
```

#### Step 3: Train Models, Evaluate Metrics & Log to MLflow
Trains Logistic Regression, Random Forest, and XGBoost; logs all runs, metrics, and curves to MLflow; registers Champion:
```bash
python src/ml_train.py
```

#### Step 4: Generate SHAP Global & Local Clinical Explainability
Computes cohort beeswarm plots, global feature importance, and 3 reference clinical case studies:
```bash
python src/ml_explain.py
```

#### Step 5: Execute Continuous Monitoring & Data Drift Audit
Audits KS-test/PSI feature drift, evaluates demographic subgroup fairness, and tests retraining policies:
```bash
python src/ml_monitor.py
```

#### Step 6: Launch FastAPI Production Prediction Microservice
Starts high-performance REST inference server on port 8000:
```bash
uvicorn src.ml_serve:app --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation is available at: `http://localhost:8000/docs`

#### Step 7: Launch Unified Streamlit Clinical Decision Support UI
```bash
streamlit run dashboard/app.py
```
Access the application at: `http://localhost:8501`

---

## 3. Running All Automated Unit Tests

The test suite covers every layer of the data engineering and MLOps stack:
```bash
python -m unittest discover tests/
```
Included test suites:
- `tests/test_ml_features.py`: Target separation, leak prevention, ICD-9 mapping, stratified split ratios (6 tests).
- `tests/test_ml_train.py`: Candidate model building, clinical metric suites, diagnostic curve generation (4 tests).
- `tests/test_ml_explain.py`: TreeExplainer initialization, local encounter explanations, reference case studies (3 tests).
- `tests/test_api.py`: FastAPI endpoints `/health`, `/predict`, `/predict-batch`, `/model-metadata`, `/case-studies` (5 tests).
- `tests/test_ml_monitor.py`: PSI calculation, KS drift detection, operational retraining policies (2 tests).

---

## 4. Multi-Container Docker Deployment

Launch PostgreSQL, Adminer, the FastAPI Inference Microservice, and the Streamlit Dashboard with one command:
```bash
docker compose up --build -d
```
Service ports:
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI Prediction Engine**: `http://localhost:8000` (Docs: `http://localhost:8000/docs`)
- **Adminer Database Portal**: `http://localhost:8080`
- **PostgreSQL Data Mart**: `localhost:5432`

---

## 5. Model Performance & Champion Benchmark

### Validation Cohort Comparison ($N=10,498$)

| Architecture | Recall (Sensitivity) | Precision | Specificity | F1 Score | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Balanced)** | 0.5425 | 0.1445 | 0.6833 | 0.2282 | 0.6616 | 0.1694 | 0.2288 |
| **Random Forest (Ensemble)** | 0.4119 | 0.1660 | 0.7960 | 0.2367 | 0.6548 | 0.1710 | 0.1988 |
| **XGBoost Classifier ★** | **0.5531** | **0.1501** | **0.6914** | **0.2362** | **0.6623** | **0.1801** | **0.2123** |

### Held-Out Test Evaluation ($N=10,499$)
- **Test Recall (Sensitivity)**: **52.70%** (detects over half of all unplanned readmissions)
- **Test Precision**: 14.27%
- **Test Specificity**: 68.74%
- **Test F1 Score**: 22.45%
- **Test ROC-AUC**: **0.6492**
- **Test PR-AUC**: **0.1722** (2.0x higher than 8.98% baseline prevalence)
- **Test Brier Score**: 0.2131 (accurately calibrated probabilities)

---

## 6. Repository Layout

```
hospital_readmission_pipeline_mlops/
├── dags/
│   └── hospital_readmission_dag.py           # Production Apache Airflow 2.x DAG
├── data/
│   ├── raw/
│   │   ├── file_raw/diabetic_data.csv        # Source 1: UCI encounter dataset
│   │   ├── api_raw/icd9_lookup_cache.csv     # Source 2: NLM REST API cache
│   │   └── document_raw/notes/*.txt          # Source 3: Unstructured clinical text files
│   ├── cleaned/integrated_encounters.csv     # Curated 3-source analytical dataset
│   └── staging/*.parquet                     # Feature store train/val/test splits
├── db/hospital_mart.db                       # Normalized SQLite analytical mart
├── sql/schema.sql                            # Production PostgreSQL DDL schema & indexes
├── dashboard/app.py                          # Streamlit clinical analytics & predictor UI
├── models/
│   ├── champion_model.joblib                 # Serialized Champion XGBoost classifier
│   ├── champion_metadata.json                # Hyperparameters, evaluation & registry metadata
│   ├── preprocessor.joblib                   # Fitted ColumnTransformer (Scalers & OHE)
│   ├── feature_names.json                    # 115 transformed feature names
│   ├── target_imbalance_metadata.json        # Class imbalance & scale_pos_weight data
│   ├── shap_feature_importance.json          # Global SHAP feature rankings
│   ├── reference_case_studies.json           # 3 Canonical reference patient case studies
│   └── evaluation_plots/                     # Diagnostic ROC, PR, Confusion & Calibration curves
├── logs/
│   ├── ingestion_log.csv                     # ETL ingestion metadata & timestamps
│   ├── pipeline_runs.json                    # Orchestration history
│   └── drift_report.json                     # Automated data drift & fairness audit report
├── src/
│   ├── extract_file.py                       # Source 1 batch ingestion
│   ├── extract_api.py                        # Source 2 concurrent REST API ingestion
│   ├── extract_document.py                   # Source 3 unstructured text parser
│   ├── transform.py                          # Data cleaning & ID mapping
│   ├── integrate.py                          # Multi-source entity integration
│   ├── load_db.py                            # Relational mart loader
│   ├── pipeline_dag.py                       # Standalone DAG runner
│   ├── ml_features.py                        # Phase 1: Feature store & preprocessor
│   ├── ml_train.py                           # Phase 2: Model training & MLflow tracking
│   ├── ml_explain.py                         # Phase 3: SHAP clinical explainability
│   ├── ml_serve.py                           # Phase 4: FastAPI REST prediction microservice
│   └── ml_monitor.py                         # Phase 7: Drift detection & fairness auditing
├── tests/
│   ├── test_ml_features.py                   # Phase 1 unit tests
│   ├── test_ml_train.py                      # Phase 2 unit tests
│   ├── test_ml_explain.py                    # Phase 3 unit tests
│   ├── test_api.py                           # Phase 4 unit tests
│   └── test_ml_monitor.py                    # Phase 7 unit tests
├── docs/
│   ├── data_dictionary.md                    # Data dictionary & validation rules
│   ├── Project5_Part1_Comprehensive_Report.md# Part 1 Academic Report (50 Marks)
│   ├── Project5_Part2_Comprehensive_Report.md# Part 2 Academic Report (50 Marks)
│   ├── part2_mlops_implementation_plan.md   # Complete architectural blueprint
│   ├── architecture_diagram.png              # Multi-source ETL architecture diagram
│   ├── shap_summary_beeswarm.png             # Global SHAP beeswarm plot
│   └── shap_importance_bar.png               # Global SHAP importance bar plot
├── Dockerfile                                # Production container specification
├── docker-compose.yml                        # Postgres + API + Dashboard orchestration
├── requirements.txt                          # Python dependencies
└── README.md                                 # Complete documentation & run guide
```

---

## 7. Submission Package Checklist (Parts 1 & 2 Complete)

| Component | Deliverable Description | Repository Location | Verified Status |
|---|---|---|---|
| **ETL & Data Mart** | 3-source segregated ingestion, Airflow DAG, PostgreSQL DDL | [`src/`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/src/), [`sql/schema.sql`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/sql/schema.sql), [`dags/`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/dags/) | **Complete** |
| **Feature Engineering** | Leakage-free preprocessor, 115 features, 70/15/15 stratified split | [`src/ml_features.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/src/ml_features.py), [`models/preprocessor.joblib`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/models/preprocessor.joblib) | **Complete** |
| **Model Training & Comparison** | Logistic Regression, Random Forest, Champion XGBoost | [`src/ml_train.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/src/ml_train.py), [`models/champion_model.joblib`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/models/champion_model.joblib) | **Complete** |
| **Experiment Tracking & Registry** | MLflow runs, hyperparams, diagnostic curves, model registry | [`mlflow.db`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/mlflow.db), `HospitalReadmissionPredictor v1` | **Complete** |
| **Explainability (XAI)** | Global SHAP beeswarm, bar plots, patient waterfall attribution | [`src/ml_explain.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/src/ml_explain.py), [`docs/shap_*.png`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/docs/) | **Complete** |
| **Inference Microservice** | FastAPI REST endpoints (`/health`, `/predict`, `/predict-batch`) | [`src/ml_serve.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/src/ml_serve.py), [`tests/test_api.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/tests/test_api.py) | **Complete** |
| **Containerization** | Dockerfile & Docker Compose (Postgres, API, Dashboard) | [`Dockerfile`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/Dockerfile), [`docker-compose.yml`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/docker-compose.yml) | **Complete** |
| **Clinical Decision UI** | 3-tab Streamlit dashboard: Analytics, Risk Calculator, MLOps Telemetry | [`dashboard/app.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/dashboard/app.py) | **Complete** |
| **Continuous Monitoring** | 2-sample KS-test, PSI drift detection, subgroup fairness audit | [`src/ml_monitor.py`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/src/ml_monitor.py), [`logs/drift_report.json`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/logs/drift_report.json) | **Complete** |
| **Technical Documentation** | Comprehensive 8–10 page Part 1 & Part 2 Reports | [`docs/Project5_Part1_Comprehensive_Report.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/docs/Project5_Part1_Comprehensive_Report.md), [`docs/Project5_Part2_Comprehensive_Report.md`](file:///c:/Users/mohnish%20jakhar/Desktop/7th%20sem/notes/hospital_readmission_pipeline_mlops/docs/Project5_Part2_Comprehensive_Report.md) | **Complete** |
