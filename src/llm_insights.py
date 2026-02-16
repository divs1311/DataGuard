"""
Module 8: LLM Integration for Automated Insights
Translates raw ML metrics into actionable executive summaries.
Uses OpenAI API if available, otherwise generates template-based insights.
"""

import json
from typing import Any


class LLMInsightGenerator:
    """Generate natural language insights from ML analysis results."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                pass

    def generate_insights(self, analysis_results: dict[str, Any]) -> str:
        """Generate executive summary from analysis results."""
        if self.client:
            return self._generate_with_llm(analysis_results)
        return self._generate_template_insights(analysis_results)

    def _build_prompt(self, results: dict[str, Any]) -> str:
        """Build the structured prompt for the LLM."""
        summary_data = self._extract_summary_data(results)
        return (
            "You are a senior data quality analyst at a major financial institution. "
            "Based on these ML analysis results, generate a concise executive summary with:\n"
            "(1) Overall data health assessment\n"
            "(2) Top 3 issues found with severity\n"
            "(3) Recommended actions prioritized by impact\n"
            "(4) Risk assessment if data is used as-is\n\n"
            f"Analysis Results:\n{json.dumps(summary_data, indent=2, default=str)}"
        )

    def _extract_summary_data(self, results: dict[str, Any]) -> dict:
        """Extract key metrics from full results for LLM context."""
        summary = {}

        # Profiling
        profile = results.get("profiling", {})
        if profile:
            summary["dataset_shape"] = profile.get("shape", {})
            summary["quality_summary"] = profile.get("quality_summary", {})
            # Top missing columns
            missing = profile.get("missing_values", {})
            top_missing = sorted(
                [(col, info["percentage"]) for col, info in missing.items() if info["count"] > 0],
                key=lambda x: x[1], reverse=True,
            )[:5]
            summary["top_missing_columns"] = [
                {"column": col, "missing_pct": pct} for col, pct in top_missing
            ]

        # Anomalies
        anomaly = results.get("anomaly_detection", {})
        if anomaly:
            summary["total_anomalies"] = anomaly.get("total_anomalies", 0)
            summary["anomaly_percentage"] = anomaly.get("anomaly_percentage", 0)

        # Classification
        classification = results.get("classification", {})
        if classification:
            best = classification.get("best_model", "")
            if best and best in classification:
                summary["best_classifier"] = best
                summary["best_classifier_f1"] = classification[best].get("f1_score", 0)

        # Regression
        regression = results.get("regression", {})
        if regression:
            best = regression.get("best_model", "")
            if best and best in regression:
                summary["best_regressor"] = best
                summary["best_regressor_r2"] = regression[best].get("r2", 0)

        # Clustering
        clustering = results.get("clustering", {})
        if clustering:
            kmeans = clustering.get("kmeans", {})
            summary["n_clusters"] = kmeans.get("optimal_k", 0)
            summary["cluster_sizes"] = kmeans.get("cluster_sizes", {})

        return summary

    def _generate_with_llm(self, results: dict[str, Any]) -> str:
        """Use OpenAI API for generating insights."""
        prompt = self._build_prompt(results)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM API error: {e}\n\n{self._generate_template_insights(results)}"

    def _generate_template_insights(self, results: dict[str, Any]) -> str:
        """Generate insights using templates when LLM API is unavailable."""
        summary = self._extract_summary_data(results)
        shape = summary.get("dataset_shape", {})
        qs = summary.get("quality_summary", {})
        n_anomalies = summary.get("total_anomalies", 0)
        anomaly_pct = summary.get("anomaly_percentage", 0)
        best_clf = summary.get("best_classifier", "N/A")
        best_f1 = summary.get("best_classifier_f1", 0)
        best_reg = summary.get("best_regressor", "N/A")
        best_r2 = summary.get("best_regressor_r2", 0)
        n_clusters = summary.get("n_clusters", 0)
        cluster_sizes = summary.get("cluster_sizes", {})
        quality_score = qs.get("overall_quality_score", 0)
        missing_pct = qs.get("overall_missing_percentage", 0)
        dup_pct = qs.get("duplicate_percentage", 0)
        top_missing = summary.get("top_missing_columns", [])

        # Determine health level
        if quality_score >= 85:
            health = "GOOD"
            health_desc = "generally healthy with minor issues"
        elif quality_score >= 65:
            health = "MODERATE"
            health_desc = "has notable quality concerns requiring attention"
        else:
            health = "POOR"
            health_desc = "has significant quality issues that must be addressed"

        # Risk assessment
        if quality_score >= 85:
            risk = "LOW — data is suitable for production use with minor cleanup"
        elif quality_score >= 65:
            risk = "MODERATE — using this data as-is carries 10-20% error risk"
        else:
            risk = "HIGH — data quality is insufficient for reliable decision-making"

        # Build missing columns text
        missing_text = ""
        if top_missing:
            cols = ", ".join([f"'{m['column']}' ({m['missing_pct']}%)" for m in top_missing[:3]])
            missing_text = f"Columns with most missing data: {cols}."

        report = f"""
{'='*60}
DATA HEALTH ASSESSMENT — EXECUTIVE SUMMARY
{'='*60}

OVERALL HEALTH: {health}
Quality Score: {quality_score}/100

Your dataset contains {shape.get('rows', 0):,} records across {shape.get('columns', 0)} columns
and is {health_desc}.

Machine learning analysis (best model: {best_clf}, F1={best_f1:.3f})
identified the following key findings:

--- TOP ISSUES ---

1. MISSING VALUES [{self._severity(missing_pct, 5, 15)}]
   {missing_pct}% of data cells are empty ({qs.get('total_missing_cells', 0):,} missing values).
   {missing_text}
   This is the strongest predictor of quality problems.

2. ANOMALIES [{self._severity(anomaly_pct, 3, 8)}]
   {n_anomalies:,} records ({anomaly_pct}%) flagged as anomalous.
   These include extreme values and records that don't match expected patterns.

3. DUPLICATES [{self._severity(dup_pct, 2, 5)}]
   {dup_pct}% of rows are duplicates.
   Suggests data collection or integration issues.

--- DATA SEGMENTS ---
Clustering identified {n_clusters} natural groups in your data:
"""
        for cid, size in sorted(cluster_sizes.items()):
            report += f"  Cluster {cid}: {size:,} rows\n"

        report += f"""
--- MODEL PERFORMANCE ---
Best classifier: {best_clf} (F1={best_f1:.4f})
Best regressor: {best_reg} (R²={best_r2:.4f})

--- RECOMMENDED ACTIONS ---
1. Investigate root cause of missing values in top affected columns
2. Review flagged anomalies for data entry errors or system issues
3. Deduplicate records and implement prevention at data source
4. Quarantine low-quality segments for manual review

--- RISK ASSESSMENT ---
{risk}

{'='*60}
"""
        return report.strip()

    @staticmethod
    def _severity(value: float, low_thresh: float, high_thresh: float) -> str:
        if value >= high_thresh:
            return "HIGH PRIORITY"
        elif value >= low_thresh:
            return "MEDIUM PRIORITY"
        return "LOW PRIORITY"
