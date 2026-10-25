import pytest

from src.predict import (
    HIGH_RISK,
    MEDIUM_RISK,
    PREDICTION_THRESHOLD,
    get_risk_band,
    get_suggested_action,
    load_pipeline,
    predict_customer,
    validate_customer,
)


# ============================================================
# Model artifact
# ============================================================

def test_saved_pipeline_is_logistic_regression(pipeline):
    # The app and API only need scikit-learn; no XGBoost at runtime.
    assert type(pipeline.named_steps["model"]).__name__ == "LogisticRegression"


def test_load_pipeline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "missing.pkl")


# ============================================================
# Validation
# ============================================================

def test_valid_customer_has_no_problems(customer):
    assert validate_customer(customer) == []


def test_missing_field(customer):
    del customer["Contract"]

    assert validate_customer(customer) == ["Missing field: Contract"]


def test_unknown_category(customer):
    customer["Contract"] = "Weekly"

    assert "Contract='Weekly'" in validate_customer(customer)[0]


def test_numeric_out_of_range(customer):
    customer["MonthlyCharges"] = 500

    assert "outside" in validate_customer(customer)[0]


def test_numeric_not_a_number(customer):
    customer["tenure"] = "three"

    assert "not numeric" in validate_customer(customer)[0]


def test_predict_rejects_invalid_input(pipeline, customer):
    customer["gender"] = "Unknown"

    with pytest.raises(ValueError, match="Invalid customer input"):
        predict_customer(pipeline, customer)


# ============================================================
# Risk band + action
# ============================================================

@pytest.mark.parametrize(
    "probability, expected",
    [
        (HIGH_RISK, "High"),
        (HIGH_RISK - 0.001, "Medium"),
        (MEDIUM_RISK, "Medium"),
        (MEDIUM_RISK - 0.001, "Low"),
    ],
)
def test_risk_band_boundaries(probability, expected):
    assert get_risk_band(probability) == expected


@pytest.mark.parametrize("band", ["High", "Medium", "Low"])
def test_every_band_has_an_action(band):
    assert get_suggested_action(band)


# ============================================================
# Prediction
# ============================================================

def test_prediction_contract(pipeline, customer):
    result = predict_customer(pipeline, customer)

    assert set(result) == {
        "churn_prediction",
        "churn_probability",
        "risk_band",
        "suggested_action",
    }
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_prediction"] == int(
        result["churn_probability"] >= PREDICTION_THRESHOLD
    )
    assert result["risk_band"] == get_risk_band(result["churn_probability"])


def test_high_risk_profile_is_flagged(pipeline, customer):
    result = predict_customer(pipeline, customer)

    assert result["churn_prediction"] == 1
    assert result["risk_band"] == "High"


def test_loyal_profile_is_low_risk(pipeline, loyal_customer):
    result = predict_customer(pipeline, loyal_customer)

    assert result["churn_prediction"] == 0
    assert result["risk_band"] == "Low"


def test_senior_citizen_encodings_agree(pipeline, customer):
    # Raw CSV uses 0/1, the app uses "No"/"Yes"; both must score the same.
    as_text = predict_customer(pipeline, {**customer, "SeniorCitizen": "Yes"})
    as_int = predict_customer(pipeline, {**customer, "SeniorCitizen": 1})

    assert as_text["churn_probability"] == pytest.approx(as_int["churn_probability"])
