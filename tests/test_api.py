import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.predict import predict_customer


@pytest.fixture(scope="module")
def client():
    # Context manager runs the lifespan, which loads the pipeline.
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_matches_inference_module(client, pipeline, customer):
    response = client.post("/predict", json=customer)

    assert response.status_code == 200
    assert response.json() == pytest.approx(predict_customer(pipeline, customer))


def test_predict_low_risk(client, loyal_customer):
    response = client.post("/predict", json=loyal_customer)

    assert response.status_code == 200
    assert response.json()["risk_band"] == "Low"


def test_unknown_category_rejected(client, customer):
    response = client.post("/predict", json={**customer, "Contract": "Weekly"})

    assert response.status_code == 422


def test_missing_field_rejected(client, customer):
    del customer["tenure"]

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_out_of_range_rejected(client, customer):
    response = client.post("/predict", json={**customer, "MonthlyCharges": 500})

    assert response.status_code == 422


def test_openapi_lists_allowed_categories(client):
    schema = client.get("/openapi.json").json()
    contract = schema["components"]["schemas"]["Customer"]["properties"]["Contract"]

    assert contract["enum"] == ["Month-to-month", "One year", "Two year"]
