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
