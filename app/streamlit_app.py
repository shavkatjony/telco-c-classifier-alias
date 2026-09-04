"""
Streamlit application for Telco Customer Churn prediction.

UI responsibility:
    Collect customer information and display results.

ML responsibility:
    src.predict handles validation, preprocessing, model inference,
    risk classification, and business recommendations.
"""

import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow Python to find the project's src package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import inference logic
from src.predict import (
    load_pipeline,
    predict_customer,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():
    """Load the trained ML pipeline once and reuse it."""

    return load_pipeline()


try:
    model = get_model()

except FileNotFoundError as error:

    st.error(str(error))

    st.info(
        "Train the model first with:\n\n"
        "`python -m src.train`"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("Telco Customer Churn Predictor")

st.markdown(
    """
Predict the probability that a customer will churn and use the result
to prioritize retention actions.

The application uses the trained machine-learning pipeline from
`models/churn_pipeline.pkl`.
"""
)


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("Customer Information")


# ------------------------------------------------------------
# Column 1
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"],
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"],
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"],
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"],
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1,
    )


# ------------------------------------------------------------
# Column 2
# ------------------------------------------------------------

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"],
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service",
        ],
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No",
        ],
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )


# ------------------------------------------------------------
# Column 3
# ------------------------------------------------------------

with col3:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service",
        ],
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year",
        ],
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"],
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

