"""
Evaluation module for Online Shopping Purchase Prediction.
Computes the 6 mandatory evaluation metrics:
- Accuracy
- AUC Score
- Precision (positive class = True)
- Recall (positive class = True)
- F1 Score (positive class = True)
- Matthews Correlation Coefficient (MCC)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "Revenue"

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def compute_prediction_scores(model, X):
    """
    Compute score/probability for positive class (True).
    Uses predict_proba if available, otherwise decision_function (LinearSVC).
    """
    positive_label = True
    classes = np.asarray(model.classes_)
    
    # Locate positive class index
    pos_idx_matches = np.where(classes == positive_label)[0]
    if len(pos_idx_matches) == 0:
        pos_idx_matches = np.where(classes == 1)[0]
    positive_index = pos_idx_matches[0]

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, positive_index]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if np.ndim(scores) == 1:
            if classes[1] != positive_label and classes[1] != 1:
                scores = -scores
        elif np.ndim(scores) == 2:
            scores = scores[:, positive_index]
    else:
        raise AttributeError("Model has neither predict_proba nor decision_function.")

    return scores


def evaluate_model(model, X_test, y_test):
    """
    Calculates all six required metrics for binary classification with Revenue=True.
    """
    positive_label = True
    y_true_bool = y_test.astype(bool)
    
    y_pred = model.predict(X_test)
    y_pred_bool = y_pred.astype(bool)
    
    scores = compute_prediction_scores(model, X_test)
    
    acc = accuracy_score(y_true_bool, y_pred_bool)
    auc = roc_auc_score(y_true_bool, scores)
    prec = precision_score(y_true_bool, y_pred_bool, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_true_bool, y_pred_bool, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true_bool, y_pred_bool, pos_label=positive_label, zero_division=0)
    mcc = matthews_corrcoef(y_true_bool, y_pred_bool)

    return {
        "Accuracy": float(acc),
        "AUC": float(auc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "MCC": float(mcc),
    }


def evaluate_all_models(models_dict, X_test, y_test):
    """
    Evaluates multiple models and returns a formatted DataFrame.
    """
    rows = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["ML Model Name"] = name
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    cols = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    return df[cols]


def get_detailed_report(model, X_test, y_test):
    """
    Computes confusion matrix and text classification report.
    Returns:
        cm: 2x2 array [[TN, FP], [FN, TP]]
        report: dict
    """
    y_true_bool = y_test.astype(bool)
    y_pred_bool = model.predict(X_test).astype(bool)
    
    cm = confusion_matrix(y_true_bool, y_pred_bool, labels=[False, True])
    report = classification_report(
        y_true_bool,
        y_pred_bool,
        target_names=["No Purchase (False)", "Purchase (True)"],
        output_dict=True,
        zero_division=0,
    )
    return cm, report


def load_all_models(model_dir="model"):
    """
    Loads all five saved trained pipeline models from disk.
    Supports both relative and absolute paths across platforms.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_dir = model_dir if os.path.isabs(model_dir) else (
        model_dir if os.path.exists(model_dir) else os.path.join(base_dir, model_dir)
    )
    
    loaded_models = {}
    for name, filename in MODEL_FILENAMES.items():
        path = os.path.join(resolved_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        loaded_models[name] = joblib.load(path)
    return loaded_models
