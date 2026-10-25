"""
Shared fixtures. Tests use the committed model and synthetic rows only,
so they run without data/raw (which is not in version control).
"""

import pytest

from src.predict import load_pipeline, sample_customer


@pytest.fixture(scope="session")
def pipeline():
    return load_pipeline()


@pytest.fixture
def customer() -> dict:
    """High-risk profile: new, month-to-month, fiber, electronic check."""

    return sample_customer()


@pytest.fixture
def loyal_customer() -> dict:
    """Low-risk profile: long tenure, two-year contract, automatic payment."""

    return {
        **sample_customer(),
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 70,
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 60.0,
        "TotalCharges": 4200.0,
    }
