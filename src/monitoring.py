"""
Module 10: Production Considerations & Monitoring
Logging, performance tracking, anomaly rate alerting, model drift detection,
and retraining triggers.
"""

import logging
import json
import time
import os
from datetime import datetime
from typing import Any


class ProductionMonitor:
    """Monitor deployed models for drift, performance degradation, and anomaly spikes."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Setup file logger
        self.logger = logging.getLogger("DataGuard")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, "dataguard.log"))
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
            self.logger.addHandler(ch)

        self.metrics_history: list[dict] = []
        self.alerts: list[dict] = []

    def log_pipeline_run(self, results: dict[str, Any], processing_time: float) -> dict:
        """Log a complete pipeline execution with metrics."""
        run_record = {
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "dataset_rows": results.get("profiling", {}).get("shape", {}).get("rows", 0),
            "quality_score": results.get("profiling", {}).get("quality_summary", {}).get(
                "overall_quality_score", 0
            ),
            "anomaly_percentage": results.get("anomaly_detection", {}).get("anomaly_percentage", 0),
            "best_classifier_f1": 0,
            "best_regressor_r2": 0,
        }

        # Extract best model metrics
        clf = results.get("classification", {})
        best_clf = clf.get("best_model", "")
        if best_clf and best_clf in clf:
            run_record["best_classifier_f1"] = clf[best_clf].get("f1_score", 0)

        reg = results.get("regression", {})
        best_reg = reg.get("best_model", "")
        if best_reg and best_reg in reg:
            run_record["best_regressor_r2"] = reg[best_reg].get("r2", 0)

        self.metrics_history.append(run_record)
        self.logger.info(f"Pipeline run complete: {json.dumps(run_record, default=str)}")

        # Check alerts
        self._check_alerts(run_record)

        # Persist metrics
        self._save_metrics()

        return run_record

    def _check_alerts(self, run_record: dict) -> None:
        """Check for anomaly spikes, performance drops, and slow processing."""
        # Alert: anomaly rate spike
        if run_record["anomaly_percentage"] > 10:
            alert = {
                "timestamp": run_record["timestamp"],
                "type": "ANOMALY_SPIKE",
                "severity": "HIGH",
                "message": (
                    f"Anomaly rate is {run_record['anomaly_percentage']}% "
                    f"(threshold: 10%). Investigate data source."
                ),
            }
            self.alerts.append(alert)
            self.logger.warning(f"ALERT: {alert['message']}")

        # Alert: model performance drop
        if len(self.metrics_history) >= 2:
            prev = self.metrics_history[-2]
            curr = run_record
            f1_drop = prev.get("best_classifier_f1", 0) - curr.get("best_classifier_f1", 0)
            if f1_drop > 0.05:
                alert = {
                    "timestamp": curr["timestamp"],
                    "type": "MODEL_DRIFT",
                    "severity": "HIGH",
                    "message": (
                        f"Classifier F1 dropped by {f1_drop:.4f} "
                        f"(from {prev['best_classifier_f1']:.4f} to {curr['best_classifier_f1']:.4f}). "
                        f"Consider retraining."
                    ),
                }
                self.alerts.append(alert)
                self.logger.warning(f"ALERT: {alert['message']}")

        # Alert: slow processing
        if run_record["processing_time_seconds"] > 60:
            alert = {
                "timestamp": run_record["timestamp"],
                "type": "SLOW_PROCESSING",
                "severity": "MEDIUM",
                "message": (
                    f"Processing took {run_record['processing_time_seconds']}s "
                    f"(threshold: 60s). Consider optimizing or scaling."
                ),
            }
            self.alerts.append(alert)
            self.logger.warning(f"ALERT: {alert['message']}")

    def _save_metrics(self) -> None:
        """Persist metrics history to JSON file."""
        path = os.path.join(self.log_dir, "metrics_history.json")
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

    def get_monitoring_report(self) -> str:
        """Return human-readable monitoring report."""
        lines = [
            "=" * 60,
            "PRODUCTION MONITORING REPORT",
            "=" * 60,
        ]

        if self.metrics_history:
            latest = self.metrics_history[-1]
            lines.extend([
                "\n--- Latest Run ---",
                f"  Timestamp: {latest['timestamp']}",
                f"  Dataset rows: {latest['dataset_rows']:,}",
                f"  Processing time: {latest['processing_time_seconds']}s",
                f"  Quality score: {latest['quality_score']}/100",
                f"  Anomaly rate: {latest['anomaly_percentage']}%",
                f"  Classifier F1: {latest['best_classifier_f1']:.4f}",
                f"  Regressor R²: {latest['best_regressor_r2']:.4f}",
            ])

            lines.append(f"\n--- History ({len(self.metrics_history)} runs) ---")
            if len(self.metrics_history) > 1:
                scores = [m["quality_score"] for m in self.metrics_history]
                lines.append(f"  Quality score trend: {' → '.join(str(s) for s in scores[-5:])}")

        if self.alerts:
            lines.append(f"\n--- Active Alerts ({len(self.alerts)}) ---")
            for alert in self.alerts[-5:]:
                lines.append(f"  [{alert['severity']}] {alert['type']}: {alert['message']}")
        else:
            lines.append("\n  No alerts. All metrics within thresholds.")

        lines.extend([
            "\n--- Deployment Guidelines ---",
            "  • Data pipeline: raw data → profiling → scoring → review queue",
            "  • Retrain models monthly or if F1 drops >5%",
            "  • Anomaly rate alert threshold: 10%",
            "  • Target prediction latency: <100ms per row",
            "=" * 60,
        ])
        return "\n".join(lines)
