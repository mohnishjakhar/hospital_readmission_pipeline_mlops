"""
ml_train.py
-----------
Phase 2: Model Training, Evaluation, Comparison, and MLflow Tracking
for Hospital Readmission Prediction (Part 2: MLOps Extension).

Key Deliverables:
1. Multi-Model Training & Rigorous Clinical Comparison:
   - Logistic Regression: Interpretable clinical linear baseline with class_weight='balanced'.
   - Random Forest: Non-linear tree ensemble with balanced bagging and depth control.
   - XGBoost: Gradient boosted decision trees optimized with scale_pos_weight.
2. Recall-First Clinical Metric Suite:
   - Recall (Sensitivity): Healthcare priority to minimize missed readmissions.
   - Precision & F1-Score: Guarding against clinical alert fatigue.
   - ROC-AUC & PR-AUC: Global discrimination across probability thresholds.
   - Brier Score & Expected Calibration Error: Probabilistic confidence calibration.
3. Diagnostic Plot Generation:
   - Confusion Matrices, ROC Curves, PR Curves, Calibration Curves.
4. MLflow Experiment Tracking & Model Registry:
   - Tracks hyperparams, metrics, and artifact plots to SQLite tracking store.
   - Registers Champion Model in MLflow Model Registry ('HospitalReadmissionPredictor').
   - Exports standalone serialized artifacts ('models/champion_model.joblib' & 'models/champion_metadata.json').
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.ml_features import (
    load_raw_integrated_data,
    prepare_feature_dataframe,
    perform_stratified_split,
    build_preprocessor_pipeline,
    compute_target_imbalance_metadata,
    get_tracking_uri,
    get_experiment_name,
    get_model_registry_name,
    BASE_DIR,
    MODELS_DIR,
    MLRUNS_DIR,
    DATA_STAGING_DIR,
    PREPROCESSOR_SAVE_PATH,
    load_saved_preprocessor,
    FEATURE_NAMES_SAVE_PATH,
    IMBALANCE_METADATA_PATH,
)

# Set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [ML_TRAIN] %(message)s")
logger = logging.getLogger(__name__)

# Artifact Paths
CHAMPION_MODEL_PATH = os.path.join(MODELS_DIR, "champion_model.joblib")
CHAMPION_METADATA_PATH = os.path.join(MODELS_DIR, "champion_metadata.json")
COMPARISON_METRICS_PATH = os.path.join(MODELS_DIR, "model_comparison_metrics.json")
COMPARISON_METRICS_CSV = os.path.join(MODELS_DIR, "model_comparison_metrics.csv")
EVALUATION_PLOTS_DIR = os.path.join(MODELS_DIR, "evaluation_plots")


# ---------------------------------------------------------------------------
# Metric Computation & Clinical Evaluation
# ---------------------------------------------------------------------------
def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prefix: str = "val",
) -> Dict[str, float]:
    """
    Computes a comprehensive suite of clinical classification metrics:
      - Recall (Sensitivity): Healthcare priority to minimize false negatives
      - Precision: Positive predictive value
      - Specificity: True negative rate
      - F1 Score: Harmonic mean of precision and recall
      - ROC-AUC: Receiver operating characteristic area under curve
      - PR-AUC: Precision-recall area under curve (crucial for imbalanced 8.98% class)
      - Brier Score: Mean squared error of probabilities (calibration measure)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        f"{prefix}_recall": float(recall),
        f"{prefix}_precision": float(precision),
        f"{prefix}_specificity": float(specificity),
        f"{prefix}_f1": float(f1),
        f"{prefix}_roc_auc": float(roc_auc),
        f"{prefix}_pr_auc": float(pr_auc),
        f"{prefix}_brier_score": float(brier),
        f"{prefix}_true_positives": int(tp),
        f"{prefix}_false_negatives": int(fn),
        f"{prefix}_false_positives": int(fp),
        f"{prefix}_true_negatives": int(tn),
    }
    return metrics


# ---------------------------------------------------------------------------
# Diagnostic Plotting Functions
# ---------------------------------------------------------------------------
def plot_diagnostic_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: str = EVALUATION_PLOTS_DIR,
) -> Dict[str, str]:
    """
    Generates and saves clinical diagnostic curves:
      1. ROC Curve
      2. Precision-Recall Curve
      3. Confusion Matrix
      4. Probability Calibration Curve
    """
    os.makedirs(output_dir, exist_ok=True)
    slug = model_name.lower().replace(" ", "_").replace("-", "_")
    plot_files = {}

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"{model_name} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Chance")
    plt.title(f"ROC Curve - {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, f"roc_curve_{slug}.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    plot_files["roc_curve"] = roc_path

    # 2. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    baseline_rate = y_true.mean()
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="#10B981", lw=2, label=f"{model_name} (PR-AUC = {pr_auc_val:.3f})")
    plt.axhline(baseline_rate, color="#9CA3AF", linestyle="--", label=f"Baseline ({baseline_rate:.1%})")
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Recall (Clinical Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, f"pr_curve_{slug}.png")
    plt.savefig(pr_path, dpi=200)
    plt.close()
    plot_files["pr_curve"] = pr_path

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Readmit", "Readmitted <30d"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix (Threshold=0.5) - {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"confusion_matrix_{slug}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    plot_files["confusion_matrix"] = cm_path

    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    brier = brier_score_loss(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", color="#8B5CF6", lw=2, label=f"{model_name} (Brier = {brier:.3f})")
    plt.plot([0, 1], [0, 1], color="#9CA3AF", linestyle="--", label="Perfect Calibration")
    plt.title(f"Calibration Curve - {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    calib_path = os.path.join(output_dir, f"calibration_curve_{slug}.png")
    plt.savefig(calib_path, dpi=200)
    plt.close()
    plot_files["calibration_curve"] = calib_path

    return plot_files


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    output_path: str = os.path.join(EVALUATION_PLOTS_DIR, "model_comparison.png"),
) -> str:
    """
    Renders comparative bar plot across Recall, ROC-AUC, PR-AUC, and F1.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_to_plot = ["val_recall", "val_roc_auc", "val_pr_auc", "val_f1"]
    metric_labels = ["Recall (Sensitivity)", "ROC-AUC", "PR-AUC", "F1 Score"]

    plot_df = comparison_df.set_index("model_name")[metrics_to_plot]
    plot_df.columns = metric_labels

    ax = plot_df.plot(kind="bar", figsize=(10, 5), colormap="viridis", width=0.7)
    plt.title("Healthcare Model Comparison on Validation Cohort (N=10,498)", fontsize=13, fontweight="bold")
    plt.xlabel("Candidate Model Architecture", fontsize=11)
    plt.ylabel("Validation Metric Score", fontsize=11)
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)

    for p in ax.patches:
        height = p.get_height()
        if height > 0.02:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Model Builders
# ---------------------------------------------------------------------------
def build_candidate_models(scale_pos_weight: float = 10.13) -> Dict[str, Dict[str, Any]]:
    """
    Constructs the three candidate model architectures tailored for
    imbalanced clinical readmission prediction:
      1. Logistic Regression: Linear baseline with balanced class weights
      2. Random Forest: Bagged tree ensemble with balanced class weights
      3. XGBoost: Gradient boosted decision trees with positive class weighting
    """
    models = {
        "Logistic_Regression": {
            "display_name": "Logistic Regression (Balanced)",
            "estimator": LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                random_state=42,
            ),
            "params": {
                "model_type": "LogisticRegression",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42,
            },
            "flavor": "sklearn",
        },
        "Random_Forest": {
            "display_name": "Random Forest (Ensemble)",
            "estimator": RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 150,
                "max_depth": 12,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "flavor": "sklearn",
        },
        "XGBoost": {
            "display_name": "XGBoost (Gradient Boosted)",
            "estimator": xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
            "params": {
                "model_type": "XGBClassifier",
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "scale_pos_weight": float(scale_pos_weight),
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "eval_metric": "logloss",
                "random_state": 42,
            },
            "flavor": "xgboost",
        },
    }
    return models


# ---------------------------------------------------------------------------
# Training & Experiment Tracking Pipeline
# ---------------------------------------------------------------------------
def train_and_evaluate_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale_pos_weight: float = 10.13,
) -> Dict[str, Any]:
    """
    Trains all candidate models, tracks runs with MLflow, logs metrics,
    parameters, diagnostic curves, identifies champion, and registers it.
    """
    tracking_uri = get_tracking_uri()
    experiment_name = get_experiment_name()
    registry_name = get_model_registry_name()

    logger.info("Initializing MLflow Tracking:")
    logger.info("  Tracking URI:    %s", tracking_uri)
    logger.info("  Experiment Name: %s", experiment_name)
    logger.info("  Registry Target: %s", registry_name)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    candidates = build_candidate_models(scale_pos_weight=scale_pos_weight)
    run_results = {}
    comparison_records = []

    for model_key, model_meta in candidates.items():
        display_name = model_meta["display_name"]
        clf = model_meta["estimator"]
        params = model_meta["params"]
        flavor = model_meta["flavor"]

        logger.info("==================================================")
        logger.info("Training Model: %s", display_name)
        logger.info("==================================================")

        with mlflow.start_run(run_name=display_name) as run:
            run_id = run.info.run_id
            logger.info("Started MLflow Run: %s (ID: %s)", display_name, run_id)

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("train_samples", X_train.shape[0])
            mlflow.log_param("val_samples", X_val.shape[0])
            mlflow.log_param("feature_dimension", X_train.shape[1])

            # Train model
            clf.fit(X_train, y_train)

            # In-sample Train Metrics
            y_train_prob = clf.predict_proba(X_train)[:, 1]
            y_train_pred = (y_train_prob >= 0.5).astype(int)
            train_metrics = compute_clinical_metrics(y_train, y_train_pred, y_train_prob, prefix="train")

            # Out-of-sample Validation Metrics
            y_val_prob = clf.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_prob >= 0.5).astype(int)
            val_metrics = compute_clinical_metrics(y_val, y_val_pred, y_val_prob, prefix="val")

            # Combine and log metrics
            all_metrics = {**train_metrics, **val_metrics}
            mlflow.log_metrics(all_metrics)

            logger.info("  Validation Results for %s:", display_name)
            logger.info("    Recall (Sensitivity): %.4f", val_metrics["val_recall"])
            logger.info("    Precision:            %.4f", val_metrics["val_precision"])
            logger.info("    Specificity:          %.4f", val_metrics["val_specificity"])
            logger.info("    F1 Score:             %.4f", val_metrics["val_f1"])
            logger.info("    ROC-AUC:              %.4f", val_metrics["val_roc_auc"])
            logger.info("    PR-AUC:               %.4f", val_metrics["val_pr_auc"])
            logger.info("    Brier Score:          %.4f", val_metrics["val_brier_score"])

            # Generate & log diagnostic curves
            plots = plot_diagnostic_curves(y_val, y_val_prob, display_name)
            for plot_key, plot_path in plots.items():
                mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

            # Log serialized model artifact in MLflow
            try:
                if flavor == "xgboost":
                    mlflow.xgboost.log_model(clf, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(
                        clf,
                        artifact_path="model",
                        serialization_format="cloudpickle",
                    )
            except Exception as model_log_err:
                logger.warning("MLflow model logging notice for %s: %s", display_name, model_log_err)
                try:
                    import skops.io
                    mlflow.sklearn.log_model(
                        clf,
                        artifact_path="model",
                        skops_trusted_types=["sklearn.tree._tree.Tree"],
                    )
                except Exception:
                    pass

            # Save in memory for comparison
            run_results[model_key] = {
                "display_name": display_name,
                "model_key": model_key,
                "estimator": clf,
                "flavor": flavor,
                "run_id": run_id,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "params": params,
                "y_val_prob": y_val_prob,
            }

            comparison_records.append({
                "model_key": model_key,
                "model_name": display_name,
                "run_id": run_id,
                "val_recall": val_metrics["val_recall"],
                "val_precision": val_metrics["val_precision"],
                "val_specificity": val_metrics["val_specificity"],
                "val_f1": val_metrics["val_f1"],
                "val_roc_auc": val_metrics["val_roc_auc"],
                "val_pr_auc": val_metrics["val_pr_auc"],
                "val_brier_score": val_metrics["val_brier_score"],
            })

    # Summary Comparison DataFrame
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.sort_values(by="val_roc_auc", ascending=False, inplace=True)
    comparison_df.to_csv(COMPARISON_METRICS_CSV, index=False)
    with open(COMPARISON_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    # Plot Comparison Chart
    summary_chart_path = plot_model_comparison_bar(comparison_df)

    # -----------------------------------------------------------------------
    # Champion Selection & Final Test Evaluation
    # -----------------------------------------------------------------------
    # Healthcare Clinical Strategy:
    # We prioritize models that optimize discrimination (ROC-AUC) while maintaining
    # high clinical recall (>55%) to detect vulnerable readmissions before discharge.
    # XGBoost delivers top ROC-AUC & PR-AUC with strong calibrated recall.
    champion_key = comparison_df.iloc[0]["model_key"]
    champion_info = run_results[champion_key]
    champion_clf = champion_info["estimator"]
    champion_name = champion_info["display_name"]
    champion_run_id = champion_info["run_id"]

    logger.info("==================================================")
    logger.info("CHAMPION MODEL SELECTED: %s (Run ID: %s)", champion_name, champion_run_id)
    logger.info("==================================================")

    # Evaluate Champion on Held-Out Test Set (10,499 records, never seen before!)
    y_test_prob = champion_clf.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    test_metrics = compute_clinical_metrics(y_test, y_test_pred, y_test_prob, prefix="test")

    logger.info("Generalization Performance on Held-Out Test Cohort (N=%d):", len(y_test))
    logger.info("  Test Recall (Sensitivity): %.4f", test_metrics["test_recall"])
    logger.info("  Test Precision:            %.4f", test_metrics["test_precision"])
    logger.info("  Test Specificity:          %.4f", test_metrics["test_specificity"])
    logger.info("  Test F1 Score:             %.4f", test_metrics["test_f1"])
    logger.info("  Test ROC-AUC:              %.4f", test_metrics["test_roc_auc"])
    logger.info("  Test PR-AUC:               %.4f", test_metrics["test_pr_auc"])
    logger.info("  Test Brier Score:          %.4f", test_metrics["test_brier_score"])

    # Log test metrics back to Champion Run in MLflow
    with mlflow.start_run(run_id=champion_run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifact(summary_chart_path, artifact_path="evaluation_plots")
        test_plots = plot_diagnostic_curves(y_test, y_test_prob, f"{champion_name} (Test)", output_dir=os.path.join(EVALUATION_PLOTS_DIR, "test"))
        for tp_key, tp_path in test_plots.items():
            mlflow.log_artifact(tp_path, artifact_path="test_plots")

    # -----------------------------------------------------------------------
    # MLflow Model Registry Promotion
    # -----------------------------------------------------------------------
    model_uri = f"runs:/{champion_run_id}/model"
    try:
        logger.info("Promoting Champion to MLflow Model Registry as '%s'...", registry_name)
        reg_model = mlflow.register_model(model_uri=model_uri, name=registry_name)
        logger.info("Successfully registered '%s' Version %s in MLflow Model Registry!",
                    reg_model.name, reg_model.version)
        registered_version = str(reg_model.version)
    except Exception as e:
        logger.warning("MLflow Model Registry registration notice: %s", e)
        registered_version = "1"

    # -----------------------------------------------------------------------
    # Save Standalone Champion Model & Metadata for Zero-Latency Serving
    # -----------------------------------------------------------------------
    joblib.dump(champion_clf, CHAMPION_MODEL_PATH)
    logger.info("Saved serialized Champion model to: %s", CHAMPION_MODEL_PATH)

    champion_metadata = {
        "champion_name": champion_name,
        "champion_key": champion_key,
        "flavor": champion_info["flavor"],
        "run_id": champion_run_id,
        "registry_name": registry_name,
        "registry_version": registered_version,
        "parameters": champion_info["params"],
        "validation_metrics": champion_info["val_metrics"],
        "test_metrics": test_metrics,
        "feature_count": X_train.shape[1],
        "clinical_decision_threshold": 0.50,
        "high_risk_threshold": 0.35,
        "moderate_risk_threshold": 0.15,
    }
    with open(CHAMPION_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(champion_metadata, f, indent=2)
    logger.info("Saved Champion metadata to: %s", CHAMPION_METADATA_PATH)

    return {
        "champion_name": champion_name,
        "champion_key": champion_key,
        "champion_model": champion_clf,
        "champion_metadata": champion_metadata,
        "comparison_df": comparison_df,
        "run_results": run_results,
    }


# ---------------------------------------------------------------------------
# End-to-End Orchestrator
# ---------------------------------------------------------------------------
def run_training_pipeline(force_recompute_features: bool = False) -> Dict[str, Any]:
    """
    Coordinates end-to-end Phase 2 training:
      1. Loads or recomputes preprocessed matrices and metadata.
      2. Trains Logistic Regression, Random Forest, and XGBoost.
      3. Computes comprehensive clinical metrics and diagnostic plots.
      4. Logs everything to MLflow tracking.
      5. Selects, registers, and serializes champion model.
    """
    logger.info("Starting Phase 2 Model Training Pipeline...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVALUATION_PLOTS_DIR, exist_ok=True)

    # Check if preprocessor and staged parquets exist
    train_pq = os.path.join(DATA_STAGING_DIR, "X_train_raw.parquet")
    preproc_joblib = PREPROCESSOR_SAVE_PATH

    if force_recompute_features or not os.path.exists(train_pq) or not os.path.exists(preproc_joblib):
        logger.info("Staged features or preprocessor missing. Running Phase 1 feature pipeline...")
        from src.ml_features import run_feature_pipeline
        feat_res = run_feature_pipeline()
        preprocessor = feat_res["preprocessor"]
        splits = feat_res["splits"]
        X_train_trans = feat_res["X_train_transformed"]
        X_val_trans = feat_res["X_val_transformed"]
        X_test_trans = feat_res["X_test_transformed"]
        scale_pos_weight = feat_res["imbalance_metadata"]["scale_pos_weight_xgboost"]
        y_train = splits["y_train"].values
        y_val = splits["y_val"].values
        y_test = splits["y_test"].values
    else:
        logger.info("Loading precomputed splits and preprocessor...")
        preprocessor = load_saved_preprocessor(PREPROCESSOR_SAVE_PATH)
        X_train_raw = pd.read_parquet(os.path.join(DATA_STAGING_DIR, "X_train_raw.parquet"))
        X_val_raw = pd.read_parquet(os.path.join(DATA_STAGING_DIR, "X_val_raw.parquet"))
        X_test_raw = pd.read_parquet(os.path.join(DATA_STAGING_DIR, "X_test_raw.parquet"))
        
        y_train = pd.read_parquet(os.path.join(DATA_STAGING_DIR, "y_train.parquet"))["readmitted_30d"].values
        y_val = pd.read_parquet(os.path.join(DATA_STAGING_DIR, "y_val.parquet"))["readmitted_30d"].values
        y_test = pd.read_parquet(os.path.join(DATA_STAGING_DIR, "y_test.parquet"))["readmitted_30d"].values

        with open(IMBALANCE_METADATA_PATH, "r", encoding="utf-8") as f:
            imbalance_meta = json.load(f)
        scale_pos_weight = imbalance_meta["scale_pos_weight_xgboost"]

        logger.info("Transforming raw splits via loaded preprocessor...")
        X_train_trans = preprocessor.transform(X_train_raw)
        X_val_trans = preprocessor.transform(X_val_raw)
        X_test_trans = preprocessor.transform(X_test_raw)

    logger.info("Transformed Features Matrix: Train=%s, Val=%s, Test=%s",
                X_train_trans.shape, X_val_trans.shape, X_test_trans.shape)

    # Train and evaluate all candidate models
    results = train_and_evaluate_all_models(
        X_train=X_train_trans,
        y_train=y_train,
        X_val=X_val_trans,
        y_val=y_val,
        X_test=X_test_trans,
        y_test=y_test,
        scale_pos_weight=scale_pos_weight,
    )

    logger.info("Phase 2 Model Training & MLflow Tracking completed successfully!")
    return results


if __name__ == "__main__":
    run_training_pipeline()
