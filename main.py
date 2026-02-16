"""
DataGuard: ML-Powered Data Quality & Predictive Analytics System
Main entry point — orchestrates all modules end-to-end.
"""

import argparse
import time
import os
import sys
import json
import warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.data_profiler import DataProfiler
from src.anomaly_detector import AnomalyDetector
from src.classifier import QualityClassifier
from src.regressor import QualityRegressor
from src.clustering import DataClusterer
from src.model_validator import ModelValidator
from src.feature_importance import FeatureInterpreter
from src.llm_insights import LLMInsightGenerator
from src.monitoring import ProductionMonitor

warnings.filterwarnings("ignore")


def run_pipeline(data_path: str, openai_key: str = None) -> dict:
    """Execute the full DataGuard analysis pipeline."""
    start_time = time.time()
    results = {}

    print("\n" + "=" * 60)
    print("  DataGuard — ML-Powered Data Quality Analysis")
    print("=" * 60)

    # ── Load Data ──
    print(f"\n[1/10] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    # ── Module 1: Data Profiling ──
    print("\n[2/10] Running automated data profiling...")
    profiler = DataProfiler(df)
    profile = profiler.run_full_profile()
    results["profiling"] = profile
    print(profiler.print_summary())

    # ── Module 2: Anomaly Detection ──
    print("\n[3/10] Running anomaly detection (Isolation Forest + DBSCAN)...")
    anomaly_detector = AnomalyDetector(contamination=0.05)
    anomaly_results = anomaly_detector.fit_detect(df)
    results["anomaly_detection"] = {
        k: v for k, v in anomaly_results.items()
        if k not in ("isolation_forest", "dbscan") or k in ("total_anomalies", "anomaly_percentage", "feature_names")
    }
    # Keep full results for internal use
    results["anomaly_detection"]["total_anomalies"] = anomaly_results["total_anomalies"]
    results["anomaly_detection"]["anomaly_percentage"] = anomaly_results["anomaly_percentage"]
    results["anomaly_detection"]["isolation_forest_anomalies"] = anomaly_results["isolation_forest"]["anomaly_count"]
    results["anomaly_detection"]["dbscan_clusters"] = anomaly_results["dbscan"]["n_clusters"]
    results["anomaly_detection"]["dbscan_noise"] = anomaly_results["dbscan"]["anomaly_count"]
    print(anomaly_detector.get_anomaly_summary())

    # ── Build feature matrix for ML ──
    print("\n[4/10] Engineering features for ML models...")
    feature_matrix = profiler.get_feature_matrix()
    combined_anomalies = anomaly_results["combined_anomaly_labels"]
    print(f"  Feature matrix: {feature_matrix.shape[0]} rows x {feature_matrix.shape[1]} features")
    print(f"  Features: {list(feature_matrix.columns)}")

    # ── Module 3: Classification Models ──
    print("\n[5/10] Training classification models...")
    classifier = QualityClassifier()
    labels = classifier.create_labels(feature_matrix, combined_anomalies)
    print(f"  Labels: {int((labels == 0).sum())} good, {int((labels == 1).sum())} bad")

    clf_results = classifier.train_and_evaluate(feature_matrix, labels)
    results["classification"] = {
        k: v for k, v in clf_results.items()
        if k not in ("confusion_matrix",)
    }
    print(classifier.get_comparison_table())

    # ── Module 4: Regression Models ──
    print("\n[6/10] Training regression models...")
    regressor = QualityRegressor()
    quality_scores = regressor.create_quality_scores(feature_matrix, combined_anomalies)
    print(f"  Quality scores: mean={quality_scores.mean():.2f}, "
          f"min={quality_scores.min():.2f}, max={quality_scores.max():.2f}")

    reg_results = regressor.train_and_evaluate(feature_matrix, quality_scores)
    results["regression"] = {}
    for k, v in reg_results.items():
        if isinstance(v, dict):
            results["regression"][k] = {
                kk: vv for kk, vv in v.items() if kk not in ("y_test", "y_pred")
            }
        else:
            results["regression"][k] = v
    print(regressor.get_comparison_table())

    # ── Module 5: Clustering ──
    print("\n[7/10] Running clustering analysis...")
    clusterer = DataClusterer()
    cluster_results = clusterer.fit(df)
    results["clustering"] = {
        "kmeans": {
            "optimal_k": cluster_results["kmeans"]["optimal_k"],
            "cluster_sizes": cluster_results["kmeans"]["cluster_sizes"],
            "inertias": cluster_results["kmeans"]["inertias"],
            "k_range": cluster_results["kmeans"]["k_range"],
        },
        "dbscan": {
            "n_clusters": cluster_results["dbscan"]["n_clusters"],
            "noise_count": cluster_results["dbscan"]["noise_count"],
            "cluster_sizes": cluster_results["dbscan"]["cluster_sizes"],
        },
        "pca": cluster_results["pca"],
    }
    # Store numpy arrays separately for dashboard use
    results["_cluster_labels_kmeans"] = cluster_results["kmeans"]["labels"]
    results["_cluster_labels_dbscan"] = cluster_results["dbscan"]["labels"]
    results["_pca_components"] = cluster_results["pca"]["components"]
    print(clusterer.get_cluster_summary())

    # ── Module 6: Model Validation ──
    print("\n[8/10] Running model validation (cross-validation & tuning)...")
    validator = ModelValidator()

    cv_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }

    from sklearn.preprocessing import StandardScaler
    X_val = StandardScaler().fit_transform(feature_matrix)
    cv_results = validator.cross_validate_all(X_val, labels, cv_models)

    # Hyperparameter tuning
    tune_results = validator.hyperparameter_tune(X_val, labels)

    # Overfitting analysis
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_val, labels, test_size=0.2, random_state=42, stratify=labels
    )
    overfit_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
    overfit_results = validator.overfitting_analysis(overfit_models, X_tr, y_tr, X_te, y_te)

    results["validation"] = {
        "cross_validation": cv_results,
        "hyperparameter_tuning": {
            name: {k: v for k, v in info.items() if k != "best_estimator"}
            for name, info in tune_results.items()
        },
        "overfitting_analysis": overfit_results,
    }
    print(validator.get_validation_report())

    # ── Module 7: Feature Importance ──
    print("\n[9/10] Analyzing feature importance & interpretability...")
    interpreter = FeatureInterpreter()
    importance_results = interpreter.analyze(
        models=classifier.models,
        feature_names=classifier.feature_names,
        X_train=classifier.X_train,
    )
    results["feature_importance"] = {}
    for model_name, info in importance_results.items():
        results["feature_importance"][model_name] = {
            k: v for k, v in info.items() if k != "partial_dependence"
        }
        if "partial_dependence" in info:
            results["feature_importance"][model_name]["partial_dependence_features"] = list(
                info["partial_dependence"].keys()
            )
    print(interpreter.get_importance_report())

    # ── Module 8: LLM Insights ──
    print("\n[10/10] Generating insights report...")
    insight_gen = LLMInsightGenerator(api_key=openai_key)
    insights = insight_gen.generate_insights(results)
    results["llm_insights"] = insights
    print(insights)

    # ── Timing ──
    elapsed = time.time() - start_time
    results["processing_time_seconds"] = round(elapsed, 2)

    # ── Module 10: Production Monitoring ──
    monitor = ProductionMonitor()
    monitor.log_pipeline_run(results, elapsed)
    print(monitor.get_monitoring_report())

    # ── Save results ──
    os.makedirs("models", exist_ok=True)
    serializable_results = _make_serializable(results)
    with open("models/analysis_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"\nResults saved to models/analysis_results.json")
    print(f"Total processing time: {elapsed:.2f}s")

    return results


def _make_serializable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {
            k: _make_serializable(v) for k, v in obj.items()
            if not (isinstance(k, str) and k.startswith("_"))
        }
    elif isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="DataGuard: ML-Powered Data Quality Analysis")
    parser.add_argument("--data", type=str, default="data/sample_data.csv",
                        help="Path to CSV file to analyze")
    parser.add_argument("--openai-key", type=str, default=None,
                        help="OpenAI API key for LLM insights (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: File not found: {args.data}")
        sys.exit(1)

    run_pipeline(args.data, args.openai_key)


if __name__ == "__main__":
    main()
