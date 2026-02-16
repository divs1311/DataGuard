"""
Module 4: Supervised Learning - Regression Models
Linear, Ridge, Lasso regression for predicting continuous quality scores (0-100).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Any


class QualityRegressor:
    """
    Predicts continuous quality scores (0-100) using regression models.
    Score = 100 - (missing_penalty + outlier_penalty + duplicate_penalty).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results: dict[str, Any] = {}
        self.feature_names = None

    def create_quality_scores(
        self,
        feature_matrix: pd.DataFrame,
        anomaly_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Compute continuous quality scores:
        Score = 100 - (missing_pct * 30 + outlier_pct * 40 + duplicate_pct * 30)
        """
        n_cols = max(feature_matrix.shape[1], 1)

        missing_pct = feature_matrix.get("missing_ratio", pd.Series(0, index=feature_matrix.index))
        outlier_pct = anomaly_labels.astype(float)
        dup_pct = feature_matrix.get("is_duplicate", pd.Series(0, index=feature_matrix.index)).astype(float)

        scores = 100 - (missing_pct * 30 + outlier_pct * 40 + dup_pct * 30)

        # Add small noise for realism
        rng = np.random.default_rng(self.random_state)
        noise = rng.normal(0, 2, size=len(scores))
        scores = np.clip(scores + noise, 0, 100).round(2)

        return scores

    def train_and_evaluate(
        self,
        feature_matrix: pd.DataFrame,
        quality_scores: np.ndarray,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Train all 3 regression models and compare."""
        self.feature_names = list(feature_matrix.columns)

        X_scaled = self.scaler.fit_transform(feature_matrix)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, quality_scores, test_size=test_size, random_state=self.random_state,
        )

        model_configs = {
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(alpha=1.0),
            "lasso_regression": Lasso(alpha=0.1, max_iter=5000),
        }

        self.results = {}
        for name, model in model_configs.items():
            model.fit(X_train, y_train)
            self.models[name] = model

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            self.results[name] = {
                "mse": round(float(mse), 4),
                "rmse": round(float(np.sqrt(mse)), 4),
                "r2": round(float(r2_score(y_test, y_pred)), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                "y_test": y_test,
                "y_pred": y_pred,
            }

            # Coefficient analysis
            if hasattr(model, "coef_"):
                coef_dict = {}
                for fname, coef in zip(self.feature_names, model.coef_):
                    coef_dict[fname] = round(float(coef), 4)
                self.results[name]["coefficients"] = coef_dict

                if hasattr(model, "intercept_"):
                    self.results[name]["intercept"] = round(float(model.intercept_), 4)

                # Lasso feature selection
                if name == "lasso_regression":
                    selected = [f for f, c in coef_dict.items() if abs(c) > 1e-6]
                    removed = [f for f, c in coef_dict.items() if abs(c) <= 1e-6]
                    self.results[name]["selected_features"] = selected
                    self.results[name]["removed_features"] = removed

        # Best model by R²
        best_name = max(
            [k for k in self.results if k != "best_model"],
            key=lambda k: self.results[k]["r2"],
        )
        self.results["best_model"] = best_name
        self.results["best_r2"] = self.results[best_name]["r2"]

        return self.results

    def get_comparison_table(self) -> str:
        """Return formatted regression model comparison table."""
        if not self.results:
            return "No results. Run train_and_evaluate() first."

        lines = [
            "=" * 70,
            "REGRESSION MODEL COMPARISON",
            "=" * 70,
            f"{'Model':<25} {'MSE':>10} {'RMSE':>10} {'R²':>10} {'MAE':>10}",
            "-" * 70,
        ]
        for name in ["linear_regression", "ridge_regression", "lasso_regression"]:
            r = self.results[name]
            lines.append(
                f"{name:<25} {r['mse']:>10.4f} {r['rmse']:>10.4f} "
                f"{r['r2']:>10.4f} {r['mae']:>10.4f}"
            )
        lines.append("-" * 70)
        lines.append(f"Best model: {self.results['best_model']} (R²={self.results['best_r2']:.4f})")

        # Lasso feature selection
        lasso = self.results.get("lasso_regression", {})
        if "selected_features" in lasso:
            lines.append(f"\nLasso selected {len(lasso['selected_features'])} features: "
                         f"{', '.join(lasso['selected_features'])}")
            if lasso["removed_features"]:
                lines.append(f"Lasso removed {len(lasso['removed_features'])} features: "
                             f"{', '.join(lasso['removed_features'])}")

        lines.append("=" * 70)
        return "\n".join(lines)
