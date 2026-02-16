"""
Module 5: Unsupervised Learning - Clustering
K-Means (with elbow method) and DBSCAN for discovering natural data groupings.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Any


class DataClusterer:
    """Find natural data groupings using K-Means and DBSCAN."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.results: dict[str, Any] = {}
        self.X_scaled = None

    def fit(self, df: pd.DataFrame, max_k: int = 10) -> dict[str, Any]:
        """Run both clustering algorithms on numeric features."""
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df = numeric_df.fillna(numeric_df.median())

        self.X_scaled = self.scaler.fit_transform(numeric_df)

        kmeans_results = self._run_kmeans(self.X_scaled, max_k)
        dbscan_results = self._run_dbscan(self.X_scaled)
        pca_results = self._run_pca(self.X_scaled)

        self.results = {
            "kmeans": kmeans_results,
            "dbscan": dbscan_results,
            "pca": pca_results,
            "feature_names": list(numeric_df.columns),
        }
        return self.results

    def _run_kmeans(self, X: np.ndarray, max_k: int) -> dict:
        """K-Means with elbow method to select optimal K."""
        inertias = []
        k_range = range(2, min(max_k + 1, len(X)))

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(X)
            inertias.append(float(km.inertia_))

        # Elbow detection: find point of maximum curvature
        optimal_k = self._find_elbow(list(k_range), inertias)

        # Fit final model with optimal K
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        labels = self.kmeans_model.fit_predict(X)

        cluster_sizes = {}
        for label in sorted(set(labels)):
            cluster_sizes[int(label)] = int((labels == label).sum())

        return {
            "optimal_k": optimal_k,
            "labels": labels,
            "inertias": inertias,
            "k_range": list(k_range),
            "cluster_sizes": cluster_sizes,
            "cluster_centers": self.kmeans_model.cluster_centers_.tolist(),
        }

    def _find_elbow(self, k_values: list, inertias: list) -> int:
        """Find elbow point using the kneedle algorithm (simplified)."""
        if len(k_values) < 3:
            return k_values[0]

        # Normalize
        x = np.array(k_values, dtype=float)
        y = np.array(inertias, dtype=float)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

        # Distance from line connecting first and last points
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        distances = []
        for i in range(len(x_norm)):
            p = np.array([x_norm[i], y_norm[i]])
            dist = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)
            distances.append(dist)

        elbow_idx = int(np.argmax(distances))
        return k_values[elbow_idx]

    def _run_dbscan(self, X: np.ndarray, eps: float = 1.5, min_samples: int = 10) -> dict:
        """DBSCAN clustering — finds arbitrary-shaped clusters."""
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan_model.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())

        cluster_sizes = {}
        for label in sorted(set(labels)):
            name = "noise" if label == -1 else f"cluster_{label}"
            cluster_sizes[name] = int((labels == label).sum())

        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "noise_count": noise_count,
            "cluster_sizes": cluster_sizes,
        }

    def _run_pca(self, X: np.ndarray, n_components: int = 2) -> dict:
        """PCA for 2D visualization of clusters."""
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        return {
            "components": X_pca,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "total_explained_variance": round(float(sum(pca.explained_variance_ratio_)), 4),
        }

    def get_cluster_summary(self) -> str:
        """Return human-readable clustering summary."""
        if not self.results:
            return "No results. Run fit() first."

        r = self.results
        lines = [
            "=" * 60,
            "CLUSTERING REPORT",
            "=" * 60,
            "",
            "--- K-Means ---",
            f"  Optimal K (elbow method): {r['kmeans']['optimal_k']}",
        ]
        for cid, size in r["kmeans"]["cluster_sizes"].items():
            lines.append(f"    Cluster {cid}: {size} rows")

        lines.extend([
            "",
            "--- DBSCAN ---",
            f"  Clusters found: {r['dbscan']['n_clusters']}",
            f"  Noise points: {r['dbscan']['noise_count']}",
        ])
        for name, size in r["dbscan"]["cluster_sizes"].items():
            lines.append(f"    {name}: {size} rows")

        lines.extend([
            "",
            "--- PCA (2D Projection) ---",
            f"  Total explained variance: {r['pca']['total_explained_variance'] * 100:.1f}%",
            f"  PC1: {r['pca']['explained_variance_ratio'][0] * 100:.1f}%",
            f"  PC2: {r['pca']['explained_variance_ratio'][1] * 100:.1f}%",
            "=" * 60,
        ])
        return "\n".join(lines)
