"""
Module 1: Automated Data Profiling
Scans datasets to understand structure, missing values, duplicates,
statistical summaries, correlations, and distribution shapes.
"""

import pandas as pd
import numpy as np
from typing import Any


class DataProfiler:
    """Comprehensive automated data profiling engine."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.profile: dict[str, Any] = {}

    def run_full_profile(self) -> dict[str, Any]:
        """Execute all profiling checks and return consolidated report."""
        self.profile = {
            "shape": self._get_shape(),
            "data_types": self._get_data_types(),
            "missing_values": self._get_missing_values(),
            "duplicates": self._get_duplicates(),
            "numeric_stats": self._get_numeric_stats(),
            "categorical_stats": self._get_categorical_stats(),
            "correlations": self._get_correlations(),
            "distributions": self._get_distributions(),
            "quality_summary": {},
        }
        self.profile["quality_summary"] = self._compute_quality_summary()
        return self.profile

    def _get_shape(self) -> dict:
        return {"rows": int(self.df.shape[0]), "columns": int(self.df.shape[1])}

    def _get_data_types(self) -> dict:
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def _get_missing_values(self) -> dict:
        missing = self.df.isnull().sum()
        total = len(self.df)
        result = {}
        for col in self.df.columns:
            count = int(missing[col])
            result[col] = {
                "count": count,
                "percentage": round(count / total * 100, 2) if total > 0 else 0,
            }
        return result

    def _get_duplicates(self) -> dict:
        dup_mask = self.df.duplicated(keep="first")
        dup_count = int(dup_mask.sum())
        return {
            "total_duplicates": dup_count,
            "percentage": round(dup_count / len(self.df) * 100, 2) if len(self.df) > 0 else 0,
            "duplicate_row_indices": self.df[dup_mask].index.tolist()[:100],  # cap for display
        }

    def _get_numeric_stats(self) -> dict:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            stats[col] = {
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "q25": round(float(series.quantile(0.25)), 4),
                "q75": round(float(series.quantile(0.75)), 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
                "zeros_count": int((series == 0).sum()),
                "negative_count": int((series < 0).sum()),
            }
        return stats

    def _get_categorical_stats(self) -> dict:
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns
        stats = {}
        for col in cat_cols:
            series = self.df[col].dropna()
            value_counts = series.value_counts()
            stats[col] = {
                "unique_count": int(series.nunique()),
                "top_values": {str(k): int(v) for k, v in value_counts.head(10).items()},
                "missing_count": int(self.df[col].isnull().sum()),
            }
        return stats

    def _get_correlations(self) -> dict:
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {}
        corr_matrix = numeric_df.corr()
        # Find high correlations (|r| > 0.7, excluding self-correlation)
        high_corr = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    high_corr.append({
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "correlation": round(float(r), 4),
                    })
        return {
            "correlation_matrix": {
                str(row): {str(col): round(float(corr_matrix.loc[row, col]), 4) for col in cols}
                for row in cols
            },
            "high_correlations": high_corr,
        }

    def _get_distributions(self) -> dict:
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        dist_info = {}
        for col in numeric_cols:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue
            skew = float(series.skew())
            if abs(skew) < 0.5:
                shape = "approximately_normal"
            elif skew > 0:
                shape = "right_skewed"
            else:
                shape = "left_skewed"
            dist_info[col] = {
                "distribution_shape": shape,
                "skewness": round(skew, 4),
                "is_normal_like": abs(skew) < 0.5,
            }
        return dist_info

    def _compute_quality_summary(self) -> dict:
        total_cells = self.df.shape[0] * self.df.shape[1]
        total_missing = int(self.df.isnull().sum().sum())
        total_dups = self.profile["duplicates"]["total_duplicates"]
        missing_pct = round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0
        dup_pct = self.profile["duplicates"]["percentage"]

        # Overall quality score: 100 - penalties
        quality_score = max(0, round(100 - (missing_pct * 2) - (dup_pct * 1.5), 2))

        return {
            "total_cells": total_cells,
            "total_missing_cells": total_missing,
            "overall_missing_percentage": missing_pct,
            "duplicate_percentage": dup_pct,
            "overall_quality_score": quality_score,
            "high_correlation_count": len(self.profile["correlations"].get("high_correlations", [])),
        }

    def get_feature_matrix(self) -> pd.DataFrame:
        """
        Build a per-row feature matrix for ML models.
        Features: missing_count, outlier indicators, numeric stats per row.
        """
        features = pd.DataFrame(index=self.df.index)

        # Per-row missing count
        features["missing_count"] = self.df.isnull().sum(axis=1).astype(int)

        # Per-row missing ratio
        features["missing_ratio"] = features["missing_count"] / self.df.shape[1]

        # Duplicate flag
        features["is_duplicate"] = self.df.duplicated(keep=False).astype(int)

        # For numeric columns: z-scores and outlier indicators
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_data = self.df[numeric_cols]
            means = numeric_data.mean()
            stds = numeric_data.std().replace(0, 1)
            z_scores = ((numeric_data - means) / stds).abs()

            features["max_zscore"] = z_scores.max(axis=1)
            features["mean_zscore"] = z_scores.mean(axis=1)
            features["outlier_flag_count"] = (z_scores > 3).sum(axis=1).astype(int)

            # Row-level numeric stats
            features["row_numeric_std"] = numeric_data.std(axis=1)
            features["row_numeric_range"] = numeric_data.max(axis=1) - numeric_data.min(axis=1)
            features["row_zeros_count"] = (numeric_data == 0).sum(axis=1).astype(int)

        # Fill any NaN features with 0
        features = features.fillna(0)

        return features

    def print_summary(self) -> str:
        """Return human-readable profiling summary."""
        if not self.profile:
            self.run_full_profile()

        p = self.profile
        lines = [
            "=" * 60,
            "DATA PROFILING REPORT",
            "=" * 60,
            f"Shape: {p['shape']['rows']} rows x {p['shape']['columns']} columns",
            "",
            "--- Missing Values ---",
        ]
        for col, info in p["missing_values"].items():
            if info["count"] > 0:
                lines.append(f"  {col}: {info['count']} ({info['percentage']}%)")

        lines.append(f"\n--- Duplicates ---")
        lines.append(f"  Total: {p['duplicates']['total_duplicates']} ({p['duplicates']['percentage']}%)")

        lines.append(f"\n--- High Correlations (|r| > 0.7) ---")
        for hc in p["correlations"].get("high_correlations", []):
            lines.append(f"  {hc['feature_1']} & {hc['feature_2']}: {hc['correlation']}")

        lines.append(f"\n--- Distributions ---")
        for col, info in p.get("distributions", {}).items():
            lines.append(f"  {col}: {info['distribution_shape']} (skew={info['skewness']})")

        qs = p["quality_summary"]
        lines.append(f"\n--- Quality Summary ---")
        lines.append(f"  Overall Quality Score: {qs['overall_quality_score']}/100")
        lines.append(f"  Total Missing: {qs['overall_missing_percentage']}%")
        lines.append(f"  Duplicates: {qs['duplicate_percentage']}%")
        lines.append("=" * 60)

        return "\n".join(lines)
