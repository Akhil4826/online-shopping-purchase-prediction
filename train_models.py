"""
Training and serialization pipeline for Online Shopping Purchase Prediction.
Implements the 5 mandatory classifiers with scikit-learn Pipeline and ColumnTransformer:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (LinearSVC)

Saves models, generates test_data.csv, and evaluates all 6 metrics.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from evaluation import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MODEL_FILENAMES,
    evaluate_all_models,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_preprocessor():
    """
    Constructs ColumnTransformer for numerical and categorical features.
    Produces dense numpy output so it is directly compatible with GaussianNB and other estimators.
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,  # Ensure dense output for GaussianNB
    )
    return preprocessor


def train_and_save_all(data_path="data/online_shoppers.csv"):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # Validate presence of expected columns
    missing_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy().astype(bool)

    print(f"Total dataset shape: {X.shape}, Target distribution:\n{y.value_counts(normalize=True)}")

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)} instances")
    print(f"Test set size: {len(X_test)} instances")

    # Save held-out test data
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv("test_data.csv", index=False)
    print("Saved held-out test dataset to test_data.csv")

    # Define model estimators
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
    }

    os.makedirs("model", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    trained_pipelines = {}

    for name, clf in classifiers.items():
        print(f"\nTraining pipeline for: {name}...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        
        # Fit pipeline on training data only
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Save model pipeline
        filename = MODEL_FILENAMES[name]
        filepath = os.path.join("model", filename)
        joblib.dump(pipeline, filepath)
        print(f"Saved pipeline to {filepath}")

    print("\n--- Evaluating all models on held-out test data ---")
    results_df = evaluate_all_models(trained_pipelines, X_test, y_test)
    results_path = os.path.join("results", "metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved evaluation metrics to {results_path}\n")

    # Display comparison table formatted nicely
    print(results_df.to_string(index=False))
    return results_df


if __name__ == "__main__":
    train_and_save_all()
