"""
Module 9: Interactive Dashboard (Streamlit)
Full-featured data quality dashboard with charts, uploads, and what-if simulator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_profiler import DataProfiler
from src.anomaly_detector import AnomalyDetector
from src.classifier import QualityClassifier
from src.regressor import QualityRegressor
from src.clustering import DataClusterer
from src.model_validator import ModelValidator
from src.feature_importance import FeatureInterpreter
from src.llm_insights import LLMInsightGenerator
from src.monitoring import ProductionMonitor

# ─── Page Config ───
st.set_page_config(
    page_title="DataGuard — ML Data Quality",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric > div {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def run_analysis(df: pd.DataFrame) -> dict:
    """Execute the full analysis pipeline and cache results."""
    results = {}
    progress = st.progress(0, text="Starting analysis...")

    # Module 1: Profiling
    progress.progress(10, text="Profiling data...")
    profiler = DataProfiler(df)
    profile = profiler.run_full_profile()
    results["profiling"] = profile
    feature_matrix = profiler.get_feature_matrix()
    results["_feature_matrix"] = feature_matrix

    # Module 2: Anomaly Detection
    progress.progress(25, text="Detecting anomalies...")
    detector = AnomalyDetector(contamination=0.05)
    anomaly_results = detector.fit_detect(df)
    results["anomaly_detection"] = anomaly_results
    combined_anomalies = anomaly_results["combined_anomaly_labels"]

    # Module 3: Classification
    progress.progress(40, text="Training classifiers...")
    classifier = QualityClassifier()
    labels = classifier.create_labels(feature_matrix, combined_anomalies)
    clf_results = classifier.train_and_evaluate(feature_matrix, labels)
    results["classification"] = clf_results
    results["_classifier"] = classifier
    results["_labels"] = labels

    # Module 4: Regression
    progress.progress(55, text="Training regressors...")
    regressor = QualityRegressor()
    quality_scores = regressor.create_quality_scores(feature_matrix, combined_anomalies)
    reg_results = regressor.train_and_evaluate(feature_matrix, quality_scores)
    results["regression"] = reg_results
    results["_quality_scores"] = quality_scores

    # Module 5: Clustering
    progress.progress(70, text="Running clustering...")
    clusterer = DataClusterer()
    cluster_results = clusterer.fit(df)
    results["clustering"] = cluster_results

    # Module 6: Validation
    progress.progress(80, text="Validating models...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    validator = ModelValidator()
    X_val = StandardScaler().fit_transform(feature_matrix)
    cv_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
    cv_results = validator.cross_validate_all(X_val, labels, cv_models)
    results["validation"] = {"cross_validation": cv_results}

    # Module 7: Feature Importance
    progress.progress(90, text="Analyzing features...")
    interpreter = FeatureInterpreter()
    importance_results = interpreter.analyze(
        models=classifier.models,
        feature_names=classifier.feature_names,
        X_train=classifier.X_train,
    )
    results["feature_importance"] = importance_results

    # Module 8: LLM Insights
    progress.progress(95, text="Generating insights...")
    insight_gen = LLMInsightGenerator()
    # Build a safe copy without numpy arrays for insights
    safe_results = {}
    for k, v in results.items():
        if not k.startswith("_"):
            safe_results[k] = v
    insights = insight_gen.generate_insights(safe_results)
    results["llm_insights"] = insights

    progress.progress(100, text="Analysis complete!")
    return results


def section_overview(results: dict):
    """Section 1: Overview Metrics."""
    st.header("📊 Overview Metrics")

    profile = results["profiling"]
    qs = profile["quality_summary"]
    anomaly = results["anomaly_detection"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Rows", f"{profile['shape']['rows']:,}")
    col2.metric("Columns", profile['shape']['columns'])
    col3.metric("Quality Score", f"{qs['overall_quality_score']}/100")
    col4.metric("Anomalies", f"{anomaly['total_anomalies']} ({anomaly['anomaly_percentage']}%)")
    col5.metric("Missing", f"{qs['overall_missing_percentage']}%")

    # Missing values breakdown
    st.subheader("Missing Values by Column")
    missing_data = {
        col: info["percentage"]
        for col, info in profile["missing_values"].items()
        if info["count"] > 0
    }
    if missing_data:
        fig = px.bar(
            x=list(missing_data.keys()),
            y=list(missing_data.values()),
            labels={"x": "Column", "y": "Missing %"},
            color=list(missing_data.values()),
            color_continuous_scale="Reds",
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Data types
    with st.expander("Data Types"):
        st.json(profile["data_types"])


def section_model_comparison(results: dict):
    """Section 2: Model Performance Comparison."""
    st.header("🤖 Model Performance Comparison")

    tab1, tab2 = st.tabs(["Classification", "Regression"])

    with tab1:
        clf = results["classification"]
        models = ["logistic_regression", "random_forest", "gradient_boosting"]
        metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

        data = []
        for model in models:
            if model in clf:
                for metric in metrics:
                    data.append({
                        "Model": model.replace("_", " ").title(),
                        "Metric": metric.replace("_", " ").title(),
                        "Value": clf[model][metric],
                    })

        if data:
            df_metrics = pd.DataFrame(data)
            fig = px.bar(
                df_metrics, x="Metric", y="Value", color="Model",
                barmode="group", title="Classification Metrics",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=450, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            best = clf.get("best_model", "")
            if best:
                st.success(f"**Best Classifier:** {best.replace('_', ' ').title()} (F1={clf['best_f1']:.4f})")

    with tab2:
        reg = results["regression"]
        reg_models = ["linear_regression", "ridge_regression", "lasso_regression"]
        reg_metrics = ["mse", "rmse", "r2", "mae"]

        data = []
        for model in reg_models:
            if model in reg:
                for metric in reg_metrics:
                    data.append({
                        "Model": model.replace("_", " ").title(),
                        "Metric": metric.upper(),
                        "Value": reg[model][metric],
                    })

        if data:
            df_metrics = pd.DataFrame(data)
            fig = px.bar(
                df_metrics, x="Metric", y="Value", color="Model",
                barmode="group", title="Regression Metrics",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

            best = reg.get("best_model", "")
            if best:
                st.success(f"**Best Regressor:** {best.replace('_', ' ').title()} (R²={reg['best_r2']:.4f})")

        # Lasso feature selection
        lasso = reg.get("lasso_regression", {})
        if "selected_features" in lasso:
            st.info(f"Lasso selected **{len(lasso['selected_features'])}** features: "
                    f"{', '.join(lasso['selected_features'])}")


def section_feature_importance(results: dict):
    """Section 3: Feature Importance Rankings."""
    st.header("🔍 Feature Importance")

    importance = results.get("feature_importance", {})

    for model_name, info in importance.items():
        if "feature_importances" in info:
            st.subheader(f"{model_name.replace('_', ' ').title()}")
            imp_data = info["feature_importances"]
            fig = px.bar(
                x=list(imp_data.values()),
                y=list(imp_data.keys()),
                orientation="h",
                labels={"x": "Importance", "y": "Feature"},
                color=list(imp_data.values()),
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=400, yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif "coefficients" in info:
            st.subheader(f"{model_name.replace('_', ' ').title()} — Coefficients")
            coef_data = info["coefficients"]
            colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in coef_data.values()]
            fig = go.Figure(go.Bar(
                x=list(coef_data.values()),
                y=list(coef_data.keys()),
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                height=400, title="Coefficients (green=positive, red=negative)",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)


def section_cluster_visualization(results: dict):
    """Section 4: Cluster Visualization (PCA 2D)."""
    st.header("🎯 Cluster Visualization")

    clustering = results.get("clustering", {})
    kmeans = clustering.get("kmeans", {})
    pca = clustering.get("pca", {})

    col1, col2 = st.columns(2)

    # K-Means clusters on PCA
    with col1:
        st.subheader("K-Means Clusters (PCA 2D)")
        if "components" in pca and "labels" in kmeans:
            pca_data = np.array(pca["components"])
            labels = np.array(kmeans["labels"])
            fig = px.scatter(
                x=pca_data[:, 0], y=pca_data[:, 1],
                color=labels.astype(str),
                labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                color_discrete_sequence=px.colors.qualitative.Set1,
                opacity=0.6,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    # DBSCAN clusters
    with col2:
        st.subheader("DBSCAN Clusters (PCA 2D)")
        dbscan = clustering.get("dbscan", {})
        if "components" in pca and "labels" in dbscan:
            pca_data = np.array(pca["components"])
            labels = np.array(dbscan["labels"])
            fig = px.scatter(
                x=pca_data[:, 0], y=pca_data[:, 1],
                color=labels.astype(str),
                labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                color_discrete_sequence=px.colors.qualitative.Dark2,
                opacity=0.6,
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

    # Elbow plot
    if "inertias" in kmeans and "k_range" in kmeans:
        st.subheader("K-Means Elbow Plot")
        fig = px.line(
            x=kmeans["k_range"], y=kmeans["inertias"],
            labels={"x": "Number of Clusters (K)", "y": "Inertia"},
            markers=True,
        )
        optimal_k = kmeans.get("optimal_k", 0)
        if optimal_k:
            idx = kmeans["k_range"].index(optimal_k) if optimal_k in kmeans["k_range"] else 0
            fig.add_vline(x=optimal_k, line_dash="dash", line_color="red",
                          annotation_text=f"Optimal K={optimal_k}")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Cluster sizes table
    if "cluster_sizes" in kmeans:
        st.subheader("K-Means Cluster Sizes")
        sizes = kmeans["cluster_sizes"]
        fig = px.pie(
            names=[f"Cluster {k}" for k in sizes.keys()],
            values=list(sizes.values()),
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


def section_anomaly_heatmap(results: dict, df: pd.DataFrame):
    """Section 5: Anomaly Heatmap."""
    st.header("🔥 Anomaly Heatmap")

    feature_matrix = results.get("_feature_matrix")
    anomaly_labels = results["anomaly_detection"]["combined_anomaly_labels"]

    if feature_matrix is not None:
        # Sample for display (max 200 rows)
        n_display = min(200, len(feature_matrix))
        sample_idx = np.random.default_rng(42).choice(len(feature_matrix), n_display, replace=False)
        sample_idx.sort()

        sample_features = feature_matrix.iloc[sample_idx]
        sample_anomalies = anomaly_labels[sample_idx]

        # Normalize features for heatmap
        normalized = (sample_features - sample_features.min()) / (sample_features.max() - sample_features.min() + 1e-10)

        fig = go.Figure(data=go.Heatmap(
            z=normalized.values,
            x=list(normalized.columns),
            y=[f"Row {i}" for i in sample_idx],
            colorscale="RdYlGn_r",
            colorbar_title="Severity",
        ))
        fig.update_layout(height=600, title=f"Feature Severity Heatmap (sample of {n_display} rows)")
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly distribution
        st.subheader("Anomaly Distribution")
        labels_series = pd.Series(anomaly_labels)
        fig = px.pie(
            names=["Normal", "Anomaly"],
            values=[int((labels_series == 0).sum()), int((labels_series == 1).sum())],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def section_regression_predictions(results: dict):
    """Section 6: Regression Predictions scatter."""
    st.header("📈 Regression: Actual vs Predicted")

    reg = results.get("regression", {})
    best_name = reg.get("best_model", "lasso_regression")
    best_result = reg.get(best_name, {})

    if "y_test" in best_result and "y_pred" in best_result:
        y_test = np.array(best_result["y_test"])
        y_pred = np.array(best_result["y_pred"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="rgba(102, 126, 234, 0.5)", size=5),
            name="Predictions",
        ))
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(color="red", dash="dash"),
            name="Perfect Prediction",
        ))
        fig.update_layout(
            height=450,
            title=f"{best_name.replace('_', ' ').title()} — Actual vs Predicted Quality Scores",
            xaxis_title="Actual Score",
            yaxis_title="Predicted Score",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Residual plot
        residuals = y_test - y_pred
        fig2 = px.histogram(residuals, nbins=50, labels={"value": "Residual"},
                            title="Residual Distribution", color_discrete_sequence=["#764ba2"])
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)


def section_llm_report(results: dict):
    """Section 7: LLM-Generated Report."""
    st.header("📝 AI-Generated Insights Report")

    insights = results.get("llm_insights", "No insights generated.")

    with st.expander("Executive Summary", expanded=True):
        st.text(insights)

    with st.expander("Technical Details"):
        # Cross-validation results
        cv = results.get("validation", {}).get("cross_validation", {})
        if cv:
            st.subheader("Cross-Validation Results (5-Fold)")
            cv_data = []
            for name, info in cv.items():
                cv_data.append({
                    "Model": name.replace("_", " ").title(),
                    "Mean F1": round(info["mean_f1"], 4),
                    "Std F1": round(info["std_f1"], 4),
                    "Min F1": round(info["min_f1"], 4),
                    "Max F1": round(info["max_f1"], 4),
                })
            st.dataframe(pd.DataFrame(cv_data), use_container_width=True)


def section_interactive_tools(results: dict):
    """Section 8: Interactive Tools (upload, what-if, export)."""
    st.header("🛠️ Interactive Tools")

    tab1, tab2, tab3 = st.tabs(["Upload CSV", "What-If Simulator", "Export Report"])

    with tab1:
        st.subheader("Upload Your Own Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded: {new_df.shape[0]} rows x {new_df.shape[1]} columns")
            st.dataframe(new_df.head(20), use_container_width=True)
            if st.button("Run Analysis on Uploaded Data"):
                with st.spinner("Analyzing..."):
                    new_results = run_analysis(new_df)
                st.session_state["results"] = new_results
                st.session_state["df"] = new_df
                st.rerun()

    with tab2:
        st.subheader("What-If Quality Score Simulator")
        st.write("Adjust parameters to see predicted quality score:")

        col1, col2, col3 = st.columns(3)
        with col1:
            missing_pct = st.slider("Missing %", 0.0, 50.0, 5.0, 0.5)
        with col2:
            outlier_pct = st.slider("Outlier %", 0.0, 30.0, 3.0, 0.5)
        with col3:
            duplicate_pct = st.slider("Duplicate %", 0.0, 20.0, 2.0, 0.5)

        predicted_score = max(0, 100 - (missing_pct * 0.3 + outlier_pct * 0.4 + duplicate_pct * 0.3) * 10)

        st.metric("Predicted Quality Score", f"{predicted_score:.1f}/100",
                   delta=f"{predicted_score - 80:.1f} vs baseline (80)")

        if predicted_score >= 85:
            st.success("Data quality is GOOD — suitable for production use.")
        elif predicted_score >= 65:
            st.warning("Data quality is MODERATE — cleanup recommended before use.")
        else:
            st.error("Data quality is POOR — significant issues must be addressed.")

    with tab3:
        st.subheader("Export Analysis Report")
        insights = results.get("llm_insights", "No insights available.")
        st.download_button(
            label="Download Report (TXT)",
            data=insights,
            file_name="dataguard_report.txt",
            mime="text/plain",
        )

        # Export JSON results
        try:
            safe = {}
            for k, v in results.items():
                if not k.startswith("_"):
                    safe[k] = v
            json_str = json.dumps(safe, indent=2, default=str)
            st.download_button(
                label="Download Full Results (JSON)",
                data=json_str,
                file_name="dataguard_results.json",
                mime="application/json",
            )
        except Exception:
            pass


def section_validation_details(results: dict):
    """Cross-validation and overfitting details."""
    st.header("✅ Model Validation Details")

    cv = results.get("validation", {}).get("cross_validation", {})
    if cv:
        st.subheader("Cross-Validation Scores per Fold")
        for name, info in cv.items():
            scores = info.get("cv_scores", [])
            if scores:
                fig = px.bar(
                    x=[f"Fold {i+1}" for i in range(len(scores))],
                    y=scores,
                    labels={"x": "Fold", "y": "F1 Score"},
                    title=f"{name.replace('_', ' ').title()} — {info['mean_f1']:.4f} ± {info['std_f1']:.4f}",
                    color_discrete_sequence=["#667eea"],
                )
                fig.update_layout(height=300, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)


# ─── Main App ───
def main():
    st.title("🛡️ DataGuard — ML-Powered Data Quality Dashboard")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        data_source = st.radio("Data Source", ["Sample Data", "Upload CSV"])

        if data_source == "Upload CSV":
            uploaded = st.file_uploader("Upload CSV", type="csv", key="sidebar_upload")
            if uploaded:
                df = pd.read_csv(uploaded)
            else:
                st.info("Upload a CSV to begin, or switch to Sample Data.")
                return
        else:
            sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_data.csv")
            if not os.path.exists(sample_path):
                st.error("Sample data not found. Run: python data/generate_data.py")
                return
            df = pd.read_csv(sample_path)

        st.write(f"**Dataset:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(5), use_container_width=True)

        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state["run_analysis"] = True

    # Run analysis
    if st.session_state.get("run_analysis"):
        if "results" not in st.session_state or "df" not in st.session_state:
            with st.spinner("Running full ML pipeline..."):
                start = time.time()
                results = run_analysis(df)
                elapsed = time.time() - start
            st.session_state["results"] = results
            st.session_state["df"] = df
            st.session_state["elapsed"] = elapsed
        else:
            results = st.session_state["results"]
            df = st.session_state["df"]
            elapsed = st.session_state.get("elapsed", 0)

        results = st.session_state["results"]
        st.sidebar.success(f"Analysis complete in {st.session_state.get('elapsed', 0):.1f}s")

        # Render all sections
        section_overview(results)
        st.markdown("---")
        section_model_comparison(results)
        st.markdown("---")
        section_feature_importance(results)
        st.markdown("---")
        section_cluster_visualization(results)
        st.markdown("---")
        section_anomaly_heatmap(results, df)
        st.markdown("---")
        section_regression_predictions(results)
        st.markdown("---")
        section_validation_details(results)
        st.markdown("---")
        section_llm_report(results)
        st.markdown("---")
        section_interactive_tools(results)
    else:
        st.info("👈 Click **Run Analysis** in the sidebar to start the ML pipeline.")

        # Show preview
        st.header("Dataset Preview")
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_data.csv")
        if os.path.exists(sample_path):
            preview = pd.read_csv(sample_path)
            st.dataframe(preview.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
