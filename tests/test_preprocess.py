import pandas as pd
import pytest

from src.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_pipeline,
    clean_and_engineer,
    load_data,
)


def raw_rows() -> pd.DataFrame:
    """Rows shaped like the raw IBM CSV (int SeniorCitizen, text TotalCharges)."""

    base = {
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.0,
        "TotalCharges": " ",
    }

    second = {
        **base,
        "customerID": "0001-TEST",
        "SeniorCitizen": 1,
        "tenure": 13,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "MonthlyCharges": 80.0,
        "TotalCharges": "1040.5",
    }

    return pd.DataFrame([base, second])


# ============================================================
# clean_and_engineer
# ============================================================

def test_drops_identifier():
    assert "customerID" not in clean_and_engineer(raw_rows()).columns


def test_blank_total_charges_become_zero():
    cleaned = clean_and_engineer(raw_rows())

    assert cleaned["TotalCharges"].tolist() == [0.0, 1040.5]


def test_senior_citizen_mapped_to_yes_no():
    cleaned = clean_and_engineer(raw_rows())

    assert cleaned["SeniorCitizen"].tolist() == ["No", "Yes"]


def test_no_service_categories_collapse_to_no():
    cleaned = clean_and_engineer(raw_rows())

    assert cleaned.loc[0, "OnlineSecurity"] == "No"
    assert cleaned.loc[1, "MultipleLines"] == "No"


def test_addon_service_count():
    cleaned = clean_and_engineer(raw_rows())

    # Row 2 has OnlineSecurity, OnlineBackup and TechSupport.
    assert cleaned["num_addon_services"].tolist() == [0, 3]


@pytest.mark.parametrize(
    "tenure, expected",
    [(0, "0-1 yr"), (12, "0-1 yr"), (13, "1-2 yr"), (48, "2-4 yr"), (72, "4-6 yr")],
)
def test_tenure_group_bins(tenure, expected):
    rows = raw_rows().assign(tenure=tenure)

    assert clean_and_engineer(rows)["tenure_group"].iloc[0] == expected


def test_does_not_mutate_input():
    rows = raw_rows()
    before = rows.copy()

    clean_and_engineer(rows)

    pd.testing.assert_frame_equal(rows, before)


# ============================================================
# Feature pipeline
# ============================================================

def test_feature_pipeline_outputs_expected_columns():
    pipeline = build_feature_pipeline()
    features = pipeline.fit_transform(raw_rows())

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    assert features.shape == (2, len(names))
    for column in NUMERIC_FEATURES:
        assert f"numeric__{column}" in names
    for column in CATEGORICAL_FEATURES:
        assert any(name.startswith(f"categorical__{column}_") for name in names)


def test_feature_pipeline_ignores_unseen_categories():
    pipeline = build_feature_pipeline().fit(raw_rows())

    unseen = raw_rows().assign(PaymentMethod="Bank transfer (automatic)")

    assert pipeline.transform(unseen).shape[0] == 2


# ============================================================
# load_data
# ============================================================

def test_load_data_splits_target(tmp_path):
    path = tmp_path / "telco.csv"
    raw_rows().assign(Churn=["No", "Yes"]).to_csv(path, index=False)

    X, y = load_data(path)

    assert "Churn" not in X.columns
    assert y.tolist() == [0, 1]


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_load_data_missing_target(tmp_path):
    path = tmp_path / "telco.csv"
    raw_rows().to_csv(path, index=False)

    with pytest.raises(ValueError, match="Churn"):
        load_data(path)
