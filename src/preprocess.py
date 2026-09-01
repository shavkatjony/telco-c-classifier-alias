"""
predict.py — load the saved pipeline and score individual customers.

This is the inference side of the project and the engine behind the Streamlit
app. Because the persisted artifact is the *entire* pipeline (cleaning +
encoding + model), scoring a raw customer record is a one-liner: hand the
pipeline a DataFrame with the original column names and it does the rest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

# ===== PROJECT PATH  ===============
#====================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET = "Churn"
ID_COL = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# FEATURE DEFINITIONS ==============
# ==================================

SERVICE_COLS = [
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_addon_services",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

# ============================================================
# RAW INPUT SCHEMA
# Single source of truth for validation and Streamlit
# ============================================================

RAW_INPUT_SCHEMA = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes", 0, 1, "0", "1"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "tenure": (0, 72),
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "MonthlyCharges": (18.0, 120.0),
    "TotalCharges": (0.0, 9000.0),
}

# LOAD DATA
# ============================================================

def load_data(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the raw CSV and separate predictors from target.

    Returns:
        X: raw feature DataFrame
        y: binary target Series where 1 = churn, 0 = stay
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    y = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .eq("Yes")
        .astype(int)
    )

    X = df.drop(columns=[TARGET])

    return X, y

# CLEANING + FEATURE ENGINEERING
# ============================================================

def clean_and_engineer(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply deterministic, row-local transformations.

    These operations do not learn statistics from the dataset,
    so they do not introduce train/test leakage.
    """

    X = X.copy()

    # --------------------------------------------------------
    # 1. Remove identifier
    # --------------------------------------------------------

    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    # --------------------------------------------------------
    # 2. TotalCharges: text -> numeric
    # --------------------------------------------------------

    X["TotalCharges"] = pd.to_numeric(
        X["TotalCharges"],
        errors="coerce",
    )

    # Blank TotalCharges values belong to tenure=0 customers.
    X["TotalCharges"] = X["TotalCharges"].fillna(0.0)

    # --------------------------------------------------------
    # 3. Standardize SeniorCitizen
    # --------------------------------------------------------

    X["SeniorCitizen"] = (
        X["SeniorCitizen"]
        .map(
            {
                0: "No",
                1: "Yes",
                "0": "No",
                "1": "Yes",
                "No": "No",
                "Yes": "Yes",
            }
        )
        .fillna("No")
    )

    # --------------------------------------------------------
    # 4. Collapse redundant service categories
    # --------------------------------------------------------

    for column in SERVICE_COLS:

        X[column] = X[column].replace(
            {
                "No internet service": "No",
                "No phone service": "No",
            }
        )

    # --------------------------------------------------------
    # 5. Number of add-on services
    # --------------------------------------------------------

    X["num_addon_services"] = (
        X[SERVICE_COLS]
        .eq("Yes")
        .sum(axis=1)
        .astype(int)
    )

    # --------------------------------------------------------
    # 6. Tenure lifecycle group
    # --------------------------------------------------------

    X["tenure_group"] = pd.cut(
        X["tenure"],
        bins=[
            -0.1,
            12,
            24,
            48,
            72,
        ],
        labels=[
            "0-1 yr",
            "1-2 yr",
            "2-4 yr",
            "4-6 yr",
        ],
    ).astype(str)

    return X

def build_preprocessor() -> ColumnTransformer:
    """
    Build the learned preprocessing transformer.

    Numerical:
        StandardScaler

    Categorical:
        OneHotEncoder

    Unknown categories:
        ignored during inference
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    
# ============================================================
# COMPLETE FEATURE PIPELINE
# ============================================================

def build_feature_pipeline() -> Pipeline:
    """
    Build the complete feature-processing pipeline.

    raw input
        ↓
    clean_and_engineer
        ↓
    scale + one-hot encode
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
        ]
    )

