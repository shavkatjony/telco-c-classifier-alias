# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn risk and translates model output into practical retention priorities.

**[Live Streamlit App →](https://shavkh.streamlit.app/)**

![Telco Churn Predictor](reports/figures/predictor_main.jpg.png)

![Predictor Result](reports/figures/predictor_out.jpg.png)
---

## Results

**Logistic Regression** was selected as the final model after comparing three classification approaches.

| Metric                   |   Test Result |
| ------------------------ | ------------: |
| ROC-AUC                  |     **0.842** |
| Churn Recall             |     **0.791** |
| True Churners Identified | **296 / 374** |

The model was selected based on **churn detection and overall discrimination rather than accuracy alone**, since missing a likely churner can be more costly in a retention setting.

---

## Business Problem

Customer churn affects recurring revenue and increases customer-acquisition pressure.

This project focuses on three questions:

* Which customers are most likely to churn?
* Which characteristics are associated with higher churn risk?
* How can predictions support a practical retention workflow?

The final application estimates churn probability, assigns a risk level, and provides a retention-oriented action.

---

## Approach

```text
Customer Data
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Train / Test Split
     ↓
Cross-Validation
     ↓
Model Comparison
     ↓
Final Evaluation
     ↓
Saved ML Pipeline
     ↓
Streamlit Application
```

### Models

* Logistic Regression
* Random Forest
* XGBoost

The training workflow uses stratified splitting and cross-validation. Preprocessing and model inference are kept together in a reusable pipeline.

---

## Model Comparison

| Model                   |   ROC-AUC | Churn Recall | Precision |    F1 |  Accuracy |
| ----------------------- | --------: | -----------: | --------: | ----: | --------: |
| **Logistic Regression** | **0.846** |    **0.797** |     0.518 | 0.628 |     0.749 |
| XGBoost                 |     0.829 |        0.700 |     0.554 | 0.618 |     0.771 |
| Random Forest           |     0.824 |        0.465 |     0.637 | 0.537 | **0.788** |

Random Forest achieved the highest accuracy, but Logistic Regression identified substantially more actual churners. For this use case, recall was therefore a more relevant decision criterion.

---

## Key Findings

The analysis identified several notable churn-risk patterns:

* **Month-to-month contracts** were strongly associated with higher churn.
* **Short-tenure customers** showed substantially higher churn risk.
* Higher-priced **fiber-optic customers** showed elevated churn risk in the analysis.
* **Electronic-check payment** and lower adoption of support/security services were associated with increased churn risk.

These are statistical associations, not evidence of causation.

![Churn Prediction Result](reports/figures/streamlit-prediction.png)

---

## Application

The Streamlit application provides:

* Customer-level churn prediction
* Churn probability
* Risk classification
* Retention-oriented recommendation

The application loads the trained pipeline from:

```text
models/churn_pipeline.pkl
```

Prediction flow:

```text
Streamlit UI
    ↓
src/predict.py
    ↓
Saved sklearn Pipeline
    ↓
Churn Probability
    ↓
Risk Level
    ↓
Business Recommendation
```

**[Try the live application →](https://shavkh.streamlit.app/)**

---

## Dataset

IBM Telco Customer Churn dataset.

| Item          | Value                                      |
| ------------- | ------------------------------------------ |
| Customers     | 7,043                                      |
| Target        | `Churn`                                    |
| Churn Rate    | 26.5%                                      |
| Main Features | Demographics, services, contracts, billing |

A data-quality issue in `TotalCharges` was handled by converting the field to numeric and treating the 11 structural blanks associated with zero-tenure customers as `$0`.

---

## Repository

```text
telco-c-classifier/
├── app/
│   └── streamlit_app.py
├── models/
│   └── churn_pipeline.pkl
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── reports/
│   ├── figures/
│   ├── findings.md
│   ├── final_model_metrics.csv
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── .gitignore
├── requirements.txt
└── README.md
```

Raw and generated datasets are excluded from version control. The trained prediction pipeline used by the application is included in the repository.

---

## Run Locally

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd telco-c-classifier

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## Limitations

* The dataset is a historical snapshot.
* Predicted probabilities represent estimated risk, not certainty.
* Statistical associations should not be interpreted as causal effects.
* Risk thresholds should be aligned with retention capacity and customer value.
* A production system would require ongoing monitoring for data drift and model performance.

---

## Technology

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Joblib · Streamlit · Jupyter · Git/GitHub**

---

## Links

**Live Application:** [shavkh.streamlit.app](https://shavkh.streamlit.app/)

**Source Code:** Add your GitHub repository URL here.
