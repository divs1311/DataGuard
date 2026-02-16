# DataGuard: ML-Powered Data Quality & Predictive Analytics System

An intelligent system that automatically analyzes any dataset and identifies quality issues using multiple machine learning techniques. It detects anomalies, predicts data quality scores, discovers natural data groupings, and generates human-readable insights.

## Features

- **Automated Data Profiling** — Scans datasets to understand structure, missing values, duplicates, distributions, and correlations
- **Anomaly Detection** — Isolation Forest + DBSCAN for comprehensive outlier detection
- **Classification Models** — Logistic Regression, Random Forest, Gradient Boosting to predict data quality labels
- **Regression Models** — Linear, Ridge, Lasso regression to predict continuous quality scores
- **Clustering** — K-Means and DBSCAN to discover natural data groupings
- **Model Validation** — Cross-validation, hyperparameter tuning, train/val/test splits
- **Feature Importance & Interpretability** — Feature rankings, coefficients, partial dependence
- **LLM Integration** — Auto-generated natural language insights from ML results
- **Interactive Dashboard** — Streamlit UI with charts, uploads, what-if simulators
- **Production Monitoring** — Logging, alerting, model drift detection

## Project Structure

```
DataGuard/
├── data/                    # Sample datasets
│   └── sample_data.csv
├── src/
│   ├── __init__.py
│   ├── data_profiler.py     # Module 1: Automated Data Profiling
│   ├── anomaly_detector.py  # Module 2: Anomaly Detection
│   ├── classifier.py        # Module 3: Classification Models
│   ├── regressor.py         # Module 4: Regression Models
│   ├── clustering.py        # Module 5: Clustering
│   ├── model_validator.py   # Module 6: Model Validation
│   ├── feature_importance.py# Module 7: Feature Importance
│   ├── llm_insights.py      # Module 8: LLM Integration
│   └── monitoring.py        # Module 10: Production Monitoring
├── dashboard/
│   └── app.py               # Module 9: Streamlit Dashboard
├── models/                  # Saved model artifacts
├── logs/                    # Application logs
├── main.py                  # CLI entry point
├── requirements.txt
├── Plan.md
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Analysis

```bash
python main.py --data data/sample_data.csv
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Modules

### Module 1: Automated Data Profiling
Scans datasets to detect missing values, data types, duplicates, statistical summaries, correlations, and distribution shapes.

### Module 2: Anomaly Detection
- **Isolation Forest**: Catches individual outlier points via random partitioning
- **DBSCAN**: Density-based detection that finds abnormal regions

### Module 3: Classification Models
Predicts binary quality labels (good/bad) using Logistic Regression, Random Forest, and Gradient Boosting with full metric comparison.

### Module 4: Regression Models
Predicts continuous quality scores (0–100) using Linear, Ridge (L2), and Lasso (L1) regression with feature selection.

### Module 5: Clustering
Discovers natural data groupings with K-Means (elbow method) and DBSCAN (density-based, automatic outlier detection).

### Module 6: Model Validation
K-fold cross-validation, GridSearchCV hyperparameter tuning, proper train/val/test splits, and overfitting prevention.

### Module 7: Feature Importance & Interpretability
Feature importance rankings, coefficient analysis, and partial dependence plots for model transparency.

### Module 8: LLM Integration
Translates raw ML metrics into actionable executive summaries using language models.

### Module 9: Interactive Dashboard
Streamlit app with overview metrics, model comparison charts, cluster visualizations, anomaly heatmaps, and CSV upload.

### Module 10: Production Monitoring
Logging, performance tracking, anomaly rate alerting, model drift detection, and retraining triggers.

## Tech Stack

- Python 3.9+
- scikit-learn (ML models)
- pandas / numpy (data manipulation)
- matplotlib / seaborn / plotly (visualization)
- streamlit (dashboard)
- openai (LLM integration)
