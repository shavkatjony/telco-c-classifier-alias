# Telco Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are most likely to churn and translates those predictions into practical customer-retention actions.

> **Headline result:** Logistic Regression was the selected model, achieving **0.842 ROC-AUC** and **0.791 recall for churners** on the held-out test set. It was selected after comparing three classification approaches while prioritizing churn detection rather than accuracy alone.

---

## 1. Business Problem

Customer churn directly affects recurring revenue and increases the need to acquire replacement customers. The business problem is therefore not simply to predict a binary label, but to identify customers with elevated churn risk early enough for the retention team to act.

This project answers three practical questions:

1. **Which customers are more likely to churn?**
2. **Which customer characteristics are most associated with churn?**
3. **How can model output be turned into a useful retention workflow?**

---

## 2. Project Objective

Build and evaluate a classification system that:

- prepares customer data for machine learning;
- compares multiple classification models;
- evaluates performance on an unseen test set;
- identifies important churn patterns;
- produces a reusable prediction pipeline; and
- exposes the model through a simple Streamlit application.

---

## 3. Dataset

This project uses the IBM sample **Telco Customer Churn** dataset.

| Item | Description |
|---|---|
| Records | 7,043 customers |
| Target | `Churn` (`Yes` / `No`) |
| Churn rate | 26.5% |
| Main feature groups | Demographics, services, contract, billing |
| Identifier removed | `customerID` |

### Data-quality issue

`TotalCharges` was stored as text and contained 11 blank values. These records correspond to customers with `tenure = 0`, so the blanks were treated as **$0** rather than dropped.

---

## 4. Machine Learning Workflow

```text
Raw customer data
       │
       ▼
Data cleaning
       │
       ▼
Feature engineering
       │
       ▼
Train / test split
       │
       ├───────────────┐
       ▼               ▼
   Training          Test
       │               │
       ▼               │
Preprocessing +        │
model training         │
       │               │
       ▼               │
Cross-validation       │
       │               │
       ▼               │
Select model           │
       │               │
       ▼               │
Fit on full train      │
       │               │
       └───────┬───────┘
               ▼
       Final test evaluation
               │
               ▼
      Saved prediction pipeline
               │
               ▼
        Streamlit application
```

The production implementation keeps cleaning/feature engineering and learned preprocessing inside the model pipeline so that training and inference use the same transformation logic.

---

## 5. Preprocessing

The preprocessing workflow includes:

- removal of the customer identifier;
- conversion of `TotalCharges` to numeric;
- treatment of structural `TotalCharges` blanks;
- normalization of the `SeniorCitizen` representation;
- consolidation of redundant service categories;
- creation of `num_addon_services`;
- creation of `tenure_group`; and
- one-hot encoding and numerical scaling inside the fitted pipeline.

The train/test split uses:

- **80% training data** — 5,634 customers
- **20% test data** — 1,409 customers
- `random_state=42`
- stratification on `Churn`

Learned transformations are fitted from training data during model training rather than being learned from the test set.

---

## 6. Models Compared

Three model families were evaluated:

| Model | Role |
|---|---|
| **Logistic Regression** | Interpretable linear baseline |
| **Random Forest** | Non-linear bagged tree ensemble |
| **Gradient Boosting** | Non-linear boosting model |

Model comparison uses stratified cross-validation on the training data, followed by final evaluation on the held-out test set.

---

## 7. Results

The project prioritizes **ROC-AUC and churn recall** because the dataset is imbalanced and a high accuracy score can hide poor churn detection.

### Cross-validation comparison

| Model | ROC-AUC | Recall (Churn) | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.846** | **0.797** | 0.518 | 0.628 | 0.749 |
| Gradient Boosting | 0.829 | 0.700 | 0.554 | 0.618 | 0.771 |
| Random Forest | 0.824 | 0.465 | 0.637 | 0.537 | **0.788** |

### Held-out test result

The selected Logistic Regression model achieved:

- **ROC-AUC:** 0.842
- **Churn recall:** 0.791
- **True churners identified:** 296 of 374
- **False negatives:** 78
- **False positives:** 298

The result illustrates why accuracy alone is not sufficient for this problem: Random Forest achieved the highest accuracy while detecting substantially fewer actual churners.

> **Key takeaway:** the best model is not necessarily the model with the highest accuracy. For a retention use case, missing a real churner can be more consequential than generating an additional false alert.

---

## 8. Key Findings

The analysis identified several strong churn signals:

1. **Month-to-month contracts** are the strongest churn-risk signal, while longer contracts are associated with lower churn risk.
2. **Short tenure** is strongly associated with churn, making the early customer lifecycle an important retention window.
3. **Fiber-optic customers at higher price points** show elevated churn risk in the analysis.
4. **Electronic-check payment** and lower use of support/security services are associated with higher churn risk.

These findings describe statistical associations; they should not be interpreted as proof of causation without further experimentation.

---

## 9. Business Recommendations

| Priority | Recommended action | Business rationale |
|---|---|---|
| High | Target high-risk month-to-month customers with contract-upgrade offers | Address the strongest churn signal |
| High | Introduce proactive onboarding during the first months of service | Reduce early-life churn |
| Medium | Review fiber pricing, support, and customer expectations | Address elevated risk in a high-value service segment |
| Medium | Promote support/security bundles and alternative payment options | Increase engagement and reduce friction |

The model should therefore be used as a **prioritization system**: identify higher-risk customers first, then apply an appropriate retention strategy.

---

## 10. Prediction Application

The project includes a Streamlit application that accepts customer information and returns:

- predicted churn outcome;
- churn probability;
- risk band; and
- suggested business action.

The application is intentionally separated from the ML logic:

```text
Streamlit UI
     ↓
src/predict.py
     ↓
Saved sklearn pipeline
     ↓
Prediction + probability
     ↓
Risk band + business action
```

This prevents the application from duplicating preprocessing or model code.

---

## 11. Repository Structure

```text
telco-churn-classifier/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   └── churn_pipeline.pkl
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── reports/
│   ├── figures/
│   ├── findings.md
│   ├── final_model_metrics.csv
│   ├── model_comparison.csv
│   └── threshold_analysis.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── __init__.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note:** The dataset and trained model are excluded from version control by `.gitignore`. A fresh clone therefore needs the raw dataset and a local model-training run before the application can be used.

---

## 12. Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd telco-churn-classifier
```

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the Telco dataset at:

```text
data/raw/telco_churn.csv
```

---

## 13. Run the Project

### Train the production pipeline

From the project root:

```bash
python -m src.train
```

This will:

1. load the raw dataset;
2. split the data into training and test sets;
3. run stratified cross-validation;
4. compare the candidate models;
5. select the strongest model by ROC-AUC;
6. fit the selected pipeline on the complete training split; and
7. save the fitted pipeline to:

```text
models/churn_pipeline.pkl
```

### Test prediction from the command line

```bash
python -m src.predict
```

### Launch the Streamlit application

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL displayed by Streamlit, normally:

```text
http://localhost:8501
```

### Explore the notebooks

```bash
jupyter notebook
```

Then open the notebooks in order:

```text
01_exploration.ipynb
02_preprocessing.ipynb
03_modeling.ipynb
04_evaluation.ipynb
```

---

## 14. Reproducibility

The project uses fixed random seeds and a deterministic train/test split for reproducibility.

The core configuration uses:

```text
random_state = 42
test_size = 0.20
stratified split
5-fold stratified cross-validation
```

The saved pipeline keeps the preprocessing and classifier together so the same transformation logic is used during inference.

---

## 15. Limitations

- The dataset is a historical snapshot and does not model when churn will happen.
- Model outputs indicate risk, not certainty or causation.
- Thresholds for High/Medium/Low risk are business decisions and should be tuned to retention capacity and customer value.
- Retention recommendations should be validated with controlled experiments such as A/B testing.
- Model performance should be monitored when new customer data becomes available.

---

## 16. Future Improvements

Possible extensions include:

- probability calibration;
- cost-sensitive threshold optimization;
- automated model monitoring;
- systematic hyperparameter tuning;
- stronger explainability with SHAP where appropriate;
- automated tests for preprocessing and inference; and
- deployment through a production API rather than only Streamlit.

---

## 17. Technologies

```text
Python
Pandas
NumPy
scikit-learn
Matplotlib
Seaborn
Joblib
Streamlit
Jupyter
Git / GitHub
```

---

## 18. Project Status

**Status: Complete end-to-end portfolio project.**

The project covers:

```text
Exploration
    → Preprocessing
    → Model Training
    → Model Comparison
    → Evaluation
    → Business Findings
    → Reusable Prediction Layer
    → Streamlit Application
```