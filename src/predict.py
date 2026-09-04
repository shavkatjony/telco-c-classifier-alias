"""
predict.py

Inference interface for the Telco churn model.

The important design decision is that the saved artifact is the complete
pipeline. Therefore this module accepts raw customer information using the
original dataset column names.

raw customer
    ↓
complete pipeline
    ↓
clean + engineer + encode + model
    ↓
churn probability
    ↓
risk band
    ↓
business action
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .preprocess import RAW_INPUT_SCHEMA


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_pipeline.pkl"
)


# ============================================================
# BUSINESS THRESHOLDS
# ============================================================

HIGH_RISK = 0.60
MEDIUM_RISK = 0.35
PREDICTION_THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

def load_pipeline(
    path: Path = DEFAULT_MODEL_PATH,
):
    """
    Load the complete trained pipeline.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run `python -m src.train` first."
        )

    return joblib.load(path)


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_customer(
    customer: dict,
) -> list[str]:
    """
    Validate raw customer input.

    Returns:
        list of validation problems.
        Empty list means valid.
    """

    problems = []

    for column, specification in RAW_INPUT_SCHEMA.items():

        if column not in customer:
            problems.append(
                f"Missing field: {column}"
            )
            continue

        value = customer[column]

        if isinstance(
            specification,
            list,
        ):

            if value not in specification:
                problems.append(
                    f"{column}={value!r} "
                    f"not in {specification}"
                )

        elif isinstance(
            specification,
            tuple,
        ):

            lower, upper = specification

            try:
                numeric_value = float(value)

            except (
                TypeError,
                ValueError,
            ):
                problems.append(
                    f"{column}={value!r} "
                    "is not numeric"
                )
                continue

            if not (
                lower
                <= numeric_value
                <= upper
            ):
                problems.append(
                    f"{column}={numeric_value} "
                    f"outside [{lower}, {upper}]"
                )

    return problems


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(
    probability: float,
) -> str:

    if probability >= HIGH_RISK:
        return "High"

    if probability >= MEDIUM_RISK:
        return "Medium"

    return "Low"


# ============================================================
# BUSINESS ACTION
# ============================================================

def get_suggested_action(
    risk_band: str,
) -> str:

    actions = {
        "High": (
            "Proactive outreach + targeted "
            "retention offer."
        ),

        "Medium": (
            "Monitor and include in the "
            "next retention campaign."
        ),

        "Low": (
            "No immediate retention action."
        ),
    }

    return actions[risk_band]


# ============================================================
# PREDICTION
# ============================================================

def predict_customer(
    pipeline,
    customer: dict,
) -> dict:
    """
    Predict churn for a single raw customer record.
    """

    # Validate input first
    problems = validate_customer(
        customer
    )

    if problems:
        raise ValueError(
            "Invalid customer input:\n"
            + "\n".join(
                f"- {problem}"
                for problem in problems
            )
        )

    # Convert dictionary to one-row DataFrame
    X = pd.DataFrame(
        [customer]
    )

    # Complete pipeline performs all transformations
    probability = float(
        pipeline.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability
        >= PREDICTION_THRESHOLD
    )

    risk_band = get_risk_band(
        probability
    )

    return {
        "churn_prediction": prediction,
        "churn_probability": probability,
        "risk_band": risk_band,
        "suggested_action": get_suggested_action(
            risk_band
        ),
    }


# ============================================================
# SAMPLE CUSTOMER
# ============================================================

def sample_customer() -> dict:
    """
    Return a sample customer for testing the inference layer.
    """

    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 285.0,
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    pipeline = load_pipeline()

    result = predict_customer(
        pipeline,
        sample_customer(),
    )

    print(
        "Sample customer prediction:"
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

        