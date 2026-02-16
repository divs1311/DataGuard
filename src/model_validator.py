"""
Module 6: Model Validation & Generalization
Cross-validation, hyperparameter tuning (GridSearchCV), 
train/val/test splits, and overfitting analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from typing import Any


class ModelValidator:
    """Comprehensive model validation with CV, tuning, and overfitting analysis."""

    def __init__(self, random_state: int = 42, cv_folds: int = 5):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.results: dict[str, Any] = {}

    def cross_validate_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        models: dict,
    ) -> dict[str, Any]:
        """Run k-fold cross-validation on all provided models."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        f1_scorer = make_scorer(f1_score, zero_division=0)

        cv_results = {}
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=f1_scorer)
            cv_results[name] = {
                "cv_scores": scores.tolist(),
                "mean_f1": round(float(scores.mean()), 4),
                "std_f1": round(float(scores.std()), 4),
                "min_f1": round(float(scores.min()), 4),
                "max_f1": round(float(scores.max()), 4),
            }

        self.results["cross_validation"] = cv_results
        return cv_results

    def hyperparameter_tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        """GridSearchCV on Random Forest and Gradient Boosting."""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        f1_scorer = make_scorer(f1_score, zero_division=0)

        # Random Forest grid
        rf_param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15],
            "min_samples_split": [2, 5, 10],
        }
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            rf_param_grid,
            scoring=f1_scorer,
            cv=cv,
            n_jobs=-1,
        )
        rf_grid.fit(X, y)

        # Gradient Boosting grid
        gb_param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=self.random_state),
            gb_param_grid,
            scoring=f1_scorer,
            cv=cv,
            n_jobs=-1,
        )
        gb_grid.fit(X, y)

        tuning_results = {
            "random_forest": {
                "best_params": rf_grid.best_params_,
                "best_cv_f1": round(float(rf_grid.best_score_), 4),
                "best_estimator": rf_grid.best_estimator_,
            },
            "gradient_boosting": {
                "best_params": gb_grid.best_params_,
                "best_cv_f1": round(float(gb_grid.best_score_), 4),
                "best_estimator": gb_grid.best_estimator_,
            },
        }

        self.results["hyperparameter_tuning"] = tuning_results
        return tuning_results

    def overfitting_analysis(
        self,
        models: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, Any]:
        """Compare train vs test performance to detect overfitting."""
        overfit_results = {}
        for name, model in models.items():
            # Refit to ensure consistency
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_f1 = round(float(f1_score(y_train, train_pred, zero_division=0)), 4)
            test_f1 = round(float(f1_score(y_test, test_pred, zero_division=0)), 4)
            gap = round(train_f1 - test_f1, 4)

            overfit_results[name] = {
                "train_f1": train_f1,
                "test_f1": test_f1,
                "gap": gap,
                "overfitting_risk": "high" if gap > 0.1 else "moderate" if gap > 0.05 else "low",
            }

        self.results["overfitting_analysis"] = overfit_results
        return overfit_results

    def get_validation_report(self) -> str:
        """Return human-readable validation report."""
        lines = [
            "=" * 70,
            "MODEL VALIDATION REPORT",
            "=" * 70,
        ]

        # Cross-validation
        cv = self.results.get("cross_validation", {})
        if cv:
            lines.append("\n--- Cross-Validation (5-Fold) ---")
            lines.append(f"{'Model':<25} {'Mean F1':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            lines.append("-" * 70)
            for name, r in cv.items():
                lines.append(
                    f"{name:<25} {r['mean_f1']:>10.4f} {r['std_f1']:>10.4f} "
                    f"{r['min_f1']:>10.4f} {r['max_f1']:>10.4f}"
                )

        # Hyperparameter tuning
        tune = self.results.get("hyperparameter_tuning", {})
        if tune:
            lines.append("\n--- Hyperparameter Tuning (GridSearchCV) ---")
            for name, r in tune.items():
                lines.append(f"  {name}:")
                lines.append(f"    Best params: {r['best_params']}")
                lines.append(f"    Best CV F1: {r['best_cv_f1']}")

        # Overfitting
        overfit = self.results.get("overfitting_analysis", {})
        if overfit:
            lines.append("\n--- Overfitting Analysis ---")
            lines.append(f"{'Model':<25} {'Train F1':>10} {'Test F1':>10} {'Gap':>10} {'Risk':>10}")
            lines.append("-" * 70)
            for name, r in overfit.items():
                lines.append(
                    f"{name:<25} {r['train_f1']:>10.4f} {r['test_f1']:>10.4f} "
                    f"{r['gap']:>10.4f} {r['overfitting_risk']:>10}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)
