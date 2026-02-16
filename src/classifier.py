"""
Module 3: Supervised Learning - Classification Models
Logistic Regression, Random Forest, Gradient Boosting for predicting
binary data quality labels (good=0, bad=1).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from typing import Any


class QualityClassifier:
    """
    Predicts binary quality labels using multiple classifiers.
    Label=1 (bad quality): anomaly OR >30% missing OR duplicate.
    Label=0 (good quality): everything else.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results: dict[str, Any] = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def create_labels(
        self,
        feature_matrix: pd.DataFrame,
        anomaly_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Generate quality labels (semi-automated):
        bad=1 if anomaly OR >30% missing ratio OR duplicate.
        """
        labels = np.zeros(len(feature_matrix), dtype=int)

        # Flagged as anomaly by combined detection
        labels[anomaly_labels == 1] = 1

        # High missing ratio (>30%)
        if "missing_ratio" in feature_matrix.columns:
            labels[feature_matrix["missing_ratio"] > 0.3] = 1

        # Duplicates
        if "is_duplicate" in feature_matrix.columns:
            labels[feature_matrix["is_duplicate"] == 1] = 1

        return labels

    def train_and_evaluate(
        self,
        feature_matrix: pd.DataFrame,
        labels: np.ndarray,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train all 3 classifiers and compare performance."""
        self.feature_names = list(feature_matrix.columns)

        X_scaled = self.scaler.fit_transform(feature_matrix)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, labels, test_size=test_size, random_state=self.random_state, stratify=labels,
        )

        model_configs = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_state
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=self.random_state,
            ),
        }

        self.results = {}
        for name, model in model_configs.items():
            model.fit(self.X_train, self.y_train)
            self.models[name] = model

            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            self.results[name] = {
                "accuracy": round(float(accuracy_score(self.y_test, y_pred)), 4),
                "precision": round(float(precision_score(self.y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(self.y_test, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(self.y_test, y_pred, zero_division=0)), 4),
                "auc_roc": round(float(roc_auc_score(self.y_test, y_proba)), 4),
                "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
                "classification_report": classification_report(
                    self.y_test, y_pred, output_dict=True, zero_division=0
                ),
            }

        # Find best model
        best_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        self.results["best_model"] = best_name
        self.results["best_f1"] = self.results[best_name]["f1_score"]

        return self.results

    def get_comparison_table(self) -> str:
        """Return formatted model comparison table."""
        if not self.results:
            return "No results. Run train_and_evaluate() first."

        lines = [
            "=" * 75,
            "CLASSIFICATION MODEL COMPARISON",
            "=" * 75,
            f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}",
            "-" * 75,
        ]
        for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            r = self.results[name]
            lines.append(
                f"{name:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                f"{r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['auc_roc']:>10.4f}"
            )
        lines.append("-" * 75)
        lines.append(f"Best model: {self.results['best_model']} (F1={self.results['best_f1']:.4f})")
        lines.append("=" * 75)
        return "\n".join(lines)
