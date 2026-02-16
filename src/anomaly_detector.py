"""
Module 2: Unsupervised Learning - Anomaly Detection
Isolation Forest + DBSCAN for comprehensive outlier detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Any


class AnomalyDetector:
    """Anomaly detection using Isolation Forest and DBSCAN."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.dbscan = None
        self.results: dict[str, Any] = {}

    def fit_detect(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run both anomaly detection algorithms on numeric features.
        Returns combined results with anomaly labels and scores.
        """
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df = numeric_df.fillna(numeric_df.median())

        X_scaled = self.scaler.fit_transform(numeric_df)

        # --- Isolation Forest ---
        if_results = self._run_isolation_forest(X_scaled)

        # --- DBSCAN ---
        dbscan_results = self._run_dbscan(X_scaled)

        # Combine: flag as anomaly if EITHER method flags it
        combined_anomaly = ((if_results["labels"] == -1) | (dbscan_results["labels"] == -1)).astype(int)

        self.results = {
            "isolation_forest": if_results,
            "dbscan": dbscan_results,
            "combined_anomaly_labels": combined_anomaly,
            "total_anomalies": int(combined_anomaly.sum()),
            "anomaly_percentage": round(float(combined_anomaly.mean()) * 100, 2),
            "feature_names": list(numeric_df.columns),
        }
        return self.results

    def _run_isolation_forest(self, X: np.ndarray) -> dict:
        """Isolation Forest: catches individual outlier points."""
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=200,
            max_samples="auto",
            random_state=self.random_state,
        )
        labels = self.isolation_forest.fit_predict(X)  # 1=normal, -1=anomaly
        scores = self.isolation_forest.decision_function(X)  # lower = more anomalous

        anomaly_count = int((labels == -1).sum())
        return {
            "labels": labels,
            "scores": scores,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(anomaly_count / len(labels) * 100, 2),
        }

    def _run_dbscan(self, X: np.ndarray, eps: float = 1.5, min_samples: int = 10) -> dict:
        """DBSCAN: density-based detection for abnormal regions."""
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan.fit_predict(X)  # -1=noise/outlier

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        anomaly_count = int((labels == -1).sum())

        cluster_sizes = {}
        for label in sorted(set(labels)):
            cluster_sizes[int(label)] = int((labels == label).sum())

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(anomaly_count / len(labels) * 100, 2),
            "cluster_sizes": cluster_sizes,
        }

    def get_anomaly_summary(self) -> str:
        """Return human-readable anomaly detection summary."""
        if not self.results:
            return "No results. Run fit_detect() first."

        r = self.results
        lines = [
            "=" * 60,
            "ANOMALY DETECTION REPORT",
            "=" * 60,
            "",
            "--- Isolation Forest ---",
            f"  Anomalies detected: {r['isolation_forest']['anomaly_count']} "
            f"({r['isolation_forest']['anomaly_percentage']}%)",
            "",
            "--- DBSCAN ---",
            f"  Clusters found: {r['dbscan']['n_clusters']}",
            f"  Noise points (anomalies): {r['dbscan']['anomaly_count']} "
            f"({r['dbscan']['anomaly_percentage']}%)",
        ]
        for cid, size in r["dbscan"]["cluster_sizes"].items():
            label = "Noise/Outliers" if cid == -1 else f"Cluster {cid}"
            lines.append(f"    {label}: {size} rows")

        lines.extend([
            "",
            "--- Combined Results ---",
            f"  Total anomalies: {r['total_anomalies']} ({r['anomaly_percentage']}%)",
            "=" * 60,
        ])
        return "\n".join(lines)
