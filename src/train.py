"""
train.py — build, cross-validate, compare, and persist churn models.

Everything a model touches is wrapped in a single scikit-learn ``Pipeline``:

        clean_and_engineer  ->  scale / one-hot  ->  classifier

so the fitted artifact on disk reproduces *every* transformation at inference
time. Training and serving cannot drift apart, because they are the same object.

Run as a script to reproduce the whole modelling stage end to end:

    python -m src.train
"""

# ========================
#  IMPORT LIBRARIES  =====
# ========================

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from .preprocess import (
    build_preprocessor,
    clean_and_engineer,
    load_data,
)

from sklearn.preprocessing import FunctionTransformer

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

PRIMARY_METRIC = "ROC-AUC"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "telco_churn.csv"
)

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_pipeline.pkl"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "model_comparison.csv"
)

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models() -> dict[str, object]:
    """
    Return the three candidate classifiers.

    Logistic Regression:
        interpretable linear baseline

    Random Forest:
        nonlinear ensemble

    HistGradientBoosting:
        gradient boosting model
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

# ============================================================
# FULL MODEL PIPELINE
# ============================================================

def build_pipeline(model) -> Pipeline:
    """
    Combine cleaning, preprocessing, and model into one object.
    """

    return Pipeline(
        steps=[
            (
                "clean",
                FunctionTransformer(
                    clean_and_engineer,
                    validate=False,
                ),
            ),
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

# ============================================================
# CROSS-VALIDATION
# ============================================================

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """
    Compare all candidate models using stratified 5-fold CV.
    """

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    rows = []

    for name, model in build_models().items():

        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        rows.append(
            {
                "Model": name,
                "Accuracy": scores[
                    "test_accuracy"
                ].mean(),

                "Precision": scores[
                    "test_precision"
                ].mean(),

                "Recall": scores[
                    "test_recall"
                ].mean(),

                "F1": scores[
                    "test_f1"
                ].mean(),

                "ROC-AUC": scores[
                    "test_roc_auc"
                ].mean(),

                "ROC-AUC Std": scores[
                    "test_roc_auc"
                ].std(),
            }
        )

    results = pd.DataFrame(rows)

    return (
        results
        .sort_values(
            by=PRIMARY_METRIC,
            ascending=False,
        )
        .reset_index(drop=True)
        .round(4)
    )

    # ============================================================
# FINAL TRAINING
# ============================================================

def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Pipeline:
    """
    Fit the selected pipeline on the complete training set.
    """

    models = build_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    pipeline = build_pipeline(
        models[model_name]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


    # FINAL TEST EVALUATION
# ============================================================

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the final pipeline on the untouched test set.
    """

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    y_prediction = (
        y_probability >= 0.50
    ).astype(int)

    return {
        "Accuracy": accuracy_score(
            y_test,
            y_prediction,
        ),

        "Precision": precision_score(
            y_test,
            y_prediction,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_test,
            y_prediction,
            zero_division=0,
        ),

        "F1": f1_score(
            y_test,
            y_prediction,
            zero_division=0,
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
    }


# ============================================================
# SAVE ARTIFACTS
# ============================================================

def save_outputs(
    pipeline: Pipeline,
    comparison: pd.DataFrame,
    final_metrics: dict,
    model_path: Path,
) -> None:
    """
    Save the final model and modeling reports.
    """

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEFAULT_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save complete pipeline
    joblib.dump(
        pipeline,
        model_path,
    )

    # Save model comparison
    comparison.to_csv(
        DEFAULT_REPORT_PATH,
        index=False,
    )

    # Save final selected-model metrics
    pd.DataFrame(
        [final_metrics]
    ).to_csv(
        PROJECT_ROOT
        / "reports"
        / "final_model_metrics.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:

    print("Loading raw data...")

    X, y = load_data(
        data_path
    )

    # --------------------------------------------------------
    # Split BEFORE learned preprocessing
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Test rows:     {len(X_test)}"
    )

    print(
        f"Churn rate:    {y.mean():.1%}"
    )

    