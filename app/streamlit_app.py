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


# BILLING INFORMATION
# ============================================================

st.header("Billing Information")

billing_col1, billing_col2 = st.columns(2)

with billing_col1:

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.0,
        max_value=120.0,
        value=70.0,
        step=1.0,
    )

with billing_col2:

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=9000.0,
        value=840.0,
        step=10.0,
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True,
)


if predict_button:

    # --------------------------------------------------------
    # Build raw customer record
    # --------------------------------------------------------

    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    try:

        result = predict_customer(
            model,
            customer,
        )

    except ValueError as error:

        st.error(str(error))
        st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.header("Prediction Result")

probability = result["churn_probability"]
risk_band = result["risk_band"]
churn = result["churn_prediction"]
action = result["suggested_action"]


    # --------------------------------------------------------
    # Main result columns
    # --------------------------------------------------------

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}",
        )

    with result_col2:

        st.metric(
            "Risk Level",
            risk_band,
        )

    with result_col3:

        prediction_text = (
            "Likely to Churn"
            if churn
            else "Likely to Stay"
        )

        st.metric(
            "Prediction",
            prediction_text,
        )


    # --------------------------------------------------------
    # Risk explanation
    # --------------------------------------------------------

    if risk_band == "High":

        st.error(
            f"High churn risk — {action}"
        )

    elif risk_band == "Medium":

        st.warning(
            f"Medium churn risk — {action}"
        )

    else:

        st.success(
            f"Low churn risk — {action}"
        )


    # --------------------------------------------------------
    # Probability visualization
    # --------------------------------------------------------

    st.subheader("Churn Probability")

    st.progress(
        int(probability * 100)
    )

    st.caption(
        f"Model probability: {probability:.1%}"
    )


    # --------------------------------------------------------
    # Business recommendation
    # --------------------------------------------------------

    st.subheader("Recommended Business Action")

    st.write(action)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Telco Customer Churn ML Project | "
    "Prediction powered by the trained sklearn pipeline"
)

