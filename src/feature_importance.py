"""
Module 7: Feature Importance & Interpretability
Feature importance rankings, coefficient analysis, partial dependence plots.
"""

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from typing import Any


class FeatureInterpreter:
    """Extract and present feature importance from trained models."""

    def __init__(self):
        self.results: dict[str, Any] = {}

    def analyze(
        self,
        models: dict,
        feature_names: list[str],
        X_train: np.ndarray = None,
    ) -> dict[str, Any]:
        """Extract feature importance from all provided models."""
        self.results = {}

        for name, model in models.items():
            model_results = {"feature_names": feature_names}

            # Tree-based feature importance
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                sorted_idx = np.argsort(importances)[::-1]
                model_results["feature_importances"] = {
                    feature_names[i]: round(float(importances[i]), 4) for i in sorted_idx
                }
                model_results["top_features"] = [
                    {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
                    for i in sorted_idx[:5]
                ]

            # Linear model coefficients
            if hasattr(model, "coef_"):
                coefs = model.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]  # For logistic regression
                sorted_idx = np.argsort(np.abs(coefs))[::-1]
                model_results["coefficients"] = {
                    feature_names[i]: round(float(coefs[i]), 4) for i in sorted_idx
                }
                model_results["top_coefficients"] = [
                    {"feature": feature_names[i], "coefficient": round(float(coefs[i]), 4)}
                    for i in sorted_idx[:5]
                ]

            # Partial Dependence (for tree-based models with training data)
            if X_train is not None and hasattr(model, "feature_importances_"):
                pd_results = self._compute_partial_dependence(model, X_train, feature_names)
                model_results["partial_dependence"] = pd_results

            self.results[name] = model_results

        return self.results

    def _compute_partial_dependence(
        self, model, X: np.ndarray, feature_names: list[str], top_n: int = 3,
    ) -> dict:
        """Compute partial dependence for top N features."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:top_n]
        else:
            top_idx = list(range(min(top_n, X.shape[1])))

        pd_data = {}
        for idx in top_idx:
            try:
                result = partial_dependence(model, X, features=[int(idx)], kind="average")
                pd_data[feature_names[idx]] = {
                    "values": result["values"][0].tolist(),
                    "avg_predictions": result["average"][0].tolist(),
                }
            except Exception:
                continue
        return pd_data

    def get_importance_report(self) -> str:
        """Return human-readable feature importance report."""
        if not self.results:
            return "No results. Run analyze() first."

        lines = [
            "=" * 60,
            "FEATURE IMPORTANCE & INTERPRETABILITY REPORT",
            "=" * 60,
        ]

        for model_name, r in self.results.items():
            lines.append(f"\n--- {model_name} ---")

            if "feature_importances" in r:
                lines.append("  Feature Importances:")
                for feat, imp in r["feature_importances"].items():
                    bar = "#" * int(imp * 50)
                    lines.append(f"    {feat:<25} {imp:.4f} {bar}")

            if "coefficients" in r:
                lines.append("  Coefficients:")
                for feat, coef in r["coefficients"].items():
                    direction = "+" if coef > 0 else ""
                    lines.append(f"    {feat:<25} {direction}{coef:.4f}")

            if "partial_dependence" in r:
                lines.append("  Partial Dependence (top features):")
                for feat, pd_info in r["partial_dependence"].items():
                    val_range = f"[{min(pd_info['values']):.2f}, {max(pd_info['values']):.2f}]"
                    pred_range = f"[{min(pd_info['avg_predictions']):.4f}, {max(pd_info['avg_predictions']):.4f}]"
                    lines.append(f"    {feat}: value range {val_range} → prediction range {pred_range}")

        lines.append("=" * 60)
        return "\n".join(lines)
