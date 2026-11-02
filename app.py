"""
Online Shopping Purchase Prediction - Streamlit Application
BITS Pilani WILP M.Tech (AIML) - Machine Learning Assignment 2

A clean, minimalist web application to demonstrate and compare five classification models
evaluated on held-out test data:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)

Note: This application loads pre-trained pipelines. It does not train or refit models.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_all_models,
    evaluate_model,
    evaluate_all_models,
    get_detailed_report,
)

# Page configuration
st.set_page_config(
    page_title="Online Shopping Purchase Prediction",
    layout="wide",
)


@st.cache_resource
def get_cached_models():
    """Load and cache pre-trained pipelines."""
    return load_all_models()


@st.cache_data
def load_default_test_data():
    """Load default held-out test dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "test_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def validate_test_dataframe(df: pd.DataFrame):
    """
    Validates uploaded test dataframe against expected schema and target column.
    Returns: (is_valid: bool, error_message: str, cleaned_df: pd.DataFrame)
    """
    if df is None or df.empty:
        return False, "The uploaded file is empty.", None

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        return (
            False,
            f"Missing required feature columns: {', '.join(missing_features)}",
            None,
        )

    if TARGET_COLUMN not in df.columns:
        return (
            False,
            f"Missing target column '{TARGET_COLUMN}' required for model evaluation.",
            None,
        )

    # Validate target column values
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    valid_bool_set = {True, False, 1, 0, "True", "False", "true", "false"}
    if not unique_targets.issubset(valid_bool_set):
        return (
            False,
            f"Target column '{TARGET_COLUMN}' must contain binary values (True/False or 1/0). Found: {unique_targets}",
            None,
        )

    df_clean = df.copy()
    if df_clean[TARGET_COLUMN].dtype == object:
        df_clean[TARGET_COLUMN] = (
            df_clean[TARGET_COLUMN]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )
    else:
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].astype(bool)

    ordered_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df_clean = df_clean[ordered_cols]
    return True, "", df_clean


def main():
    # 1. Title and description
    st.title("Online Shopping Purchase Prediction")
    st.caption("BITS Pilani WILP M.Tech (AIML) — Machine Learning Assignment 2")
    st.markdown(
        "This application evaluates five machine learning classifiers on e-commerce session data "
        "to predict whether a visitor will make a purchase (`Revenue = True`). "
        "Models are evaluated on held-out test data without retraining."
    )
    st.divider()

    # 2. Dataset and Upload Section
    st.subheader("1. Test Dataset")
    col_input1, col_input2 = st.columns([1, 2])

    with col_input1:
        data_source = st.radio(
            "Select evaluation data source:",
            ("Default Held-Out Test Data (2,466 records)", "Upload Custom Test CSV"),
            index=0,
        )

    test_df = None
    with col_input2:
        if data_source == "Default Held-Out Test Data (2,466 records)":
            test_df = load_default_test_data()
            if test_df is not None:
                st.success(f"Loaded held-out test dataset ({len(test_df)} instances, 17 features + Revenue).")
            else:
                st.error("Default test_data.csv not found in repository.")
        else:
            uploaded_file = st.file_uploader(
                "Upload a test CSV file (must include all 17 features and Revenue target column):",
                type=["csv"],
            )
            if uploaded_file is not None:
                try:
                    raw_df = pd.read_csv(uploaded_file)
                    is_valid, err_msg, cleaned_df = validate_test_dataframe(raw_df)
                    if is_valid:
                        test_df = cleaned_df
                        st.success(f"Uploaded CSV validated successfully ({len(test_df)} instances).")
                    else:
                        st.error(f"Validation Error: {err_msg}")
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

    if test_df is None or test_df.empty:
        st.info("Please provide a valid test dataset to view model evaluation results.")
        return

    # Load models
    try:
        models = get_cached_models()
    except Exception as e:
        st.error(f"Error loading trained models: {e}")
        return

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(bool)

    st.divider()

    # 3. Model Comparison Table
    st.subheader("2. Model Performance Comparison")
    st.markdown(
        "Evaluation results for all five classifiers on the active test dataset. "
        "All metrics are computed using the positive purchase class (`Revenue = True`)."
    )

    metrics_df = evaluate_all_models(models, X_test, y_test)

    # Clean formatted display table
    formatted_table = metrics_df.copy()
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        formatted_table[col] = formatted_table[col].apply(lambda v: f"{v:.4f}")

    st.dataframe(formatted_table, use_container_width=True, hide_index=True)

    st.caption(
        "Note on Class Imbalance: In this dataset, ~15.5% of sessions result in purchases. "
        "Decision Tree achieves the highest F1 Score and MCC, indicating the most balanced identification "
        "of actual purchasers. Support Vector Machine achieves the highest Precision and AUC."
    )

    st.divider()

    # 4. Model Selection & Individual Model Details
    st.subheader("3. Model Deep Dive")

    selected_model_name = st.selectbox(
        "Select a classifier to inspect:",
        options=list(models.keys()),
        index=0,
    )

    selected_model = models[selected_model_name]
    selected_metrics = evaluate_model(selected_model, X_test, y_test)
    cm, report = get_detailed_report(selected_model, X_test, y_test)

    # Show metrics in a clean summary table
    st.markdown(f"**Metrics for {selected_model_name}:**")
    metric_cols = st.columns(6)
    metric_cols[0].metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
    metric_cols[1].metric("AUC", f"{selected_metrics['AUC']:.4f}")
    metric_cols[2].metric("Precision", f"{selected_metrics['Precision']:.4f}")
    metric_cols[3].metric("Recall", f"{selected_metrics['Recall']:.4f}")
    metric_cols[4].metric("F1 Score", f"{selected_metrics['F1']:.4f}")
    metric_cols[5].metric("MCC", f"{selected_metrics['MCC']:.4f}")

    # Confusion Matrix and Classification Report
    st.write("")
    diag_col1, diag_col2 = st.columns([1, 1])

    with diag_col1:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Purchase", "Purchase"],
            yticklabels=["No Purchase", "Purchase"],
            cbar=False,
            ax=ax,
            annot_kws={"size": 11},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"{selected_model_name}", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()
        st.caption(f"True Negatives: {tn} | False Positives: {fp} | False Negatives: {fn} | True Positives: {tp}")

    with diag_col2:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(report).transpose()
        formatted_report = report_df.copy()
        for col in ["precision", "recall", "f1-score"]:
            formatted_report[col] = formatted_report[col].apply(lambda v: f"{v:.4f}")
        formatted_report["support"] = formatted_report["support"].apply(lambda v: f"{int(v)}")
        st.dataframe(formatted_report, use_container_width=True)

    st.divider()

    # 5. Dataset Information
    with st.expander("Dataset Details & Feature Information"):
        st.markdown(
            """
            - **Benchmark Dataset:** UCI Online Shoppers Purchasing Intention Dataset (ID: 468)
            - **Total Dataset Size:** 12,330 sessions
            - **Numerical Features (10):** Administrative, Administrative_Duration, Informational, Informational_Duration,
              ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay.
            - **Categorical Features (7):** Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend.
            - **Target Column:** Revenue (`True` = Purchase, `False` = No Purchase)
            - **Evaluation Split:** Stratified 80% training / 20% testing split (`random_state=42`).
            """
        )


if __name__ == "__main__":
    main()
