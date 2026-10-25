"""
FastAPI service for the Telco churn model.

Same inference path as the Streamlit app:

HTTP request
    ↓
pydantic schema (built from RAW_INPUT_SCHEMA)
    ↓
src/predict.py
    ↓
churn probability + risk band + business action

Run from the project root:

    uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.predict import (
    load_pipeline,
    predict_customer,
    sample_customer,
)
from src.preprocess import RAW_INPUT_SCHEMA


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

def _schema_field(specification):
    """
    Translate one RAW_INPUT_SCHEMA entry into a pydantic field.

    list  -> allowed categories
    tuple -> numeric range, typed by its bounds (int for tenure)
    """

    if isinstance(specification, tuple):
        lower, upper = specification
        return (type(lower), Field(ge=lower, le=upper))

    return (Literal[tuple(specification)], ...)


# Generated from the same dict that validates Streamlit input,
# so the API cannot accept a field the model was not trained on.
Customer = create_model(
    "Customer",
    __config__=ConfigDict(
        json_schema_extra={"examples": [sample_customer()]},
    ),
    **{
        column: _schema_field(specification)
        for column, specification in RAW_INPUT_SCHEMA.items()
    },
)


class Prediction(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_band: Literal["Low", "Medium", "High"]
    suggested_action: str


# ============================================================
# APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline once at startup; fail fast if it is missing."""

    app.state.pipeline = load_pipeline()
    yield


app = FastAPI(
    title="Telco Churn API",
    description="Churn probability, risk band and retention action.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> dict:

    try:
        return predict_customer(
            app.state.pipeline,
            customer.model_dump(),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )
