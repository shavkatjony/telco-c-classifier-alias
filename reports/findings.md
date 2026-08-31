# Telco Customer Churn — Findings

## 1. Business Problem

Customer churn is an important business problem because losing existing
customers can reduce recurring revenue and increase the need for customer
acquisition.

This project develops a machine learning classification model to identify
customers who are more likely to churn and support targeted retention
decisions.

---

## 2. Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains **7,043 customer records**.

The target variable is:

- `Churn` — whether the customer left the company

The `customerID` column was removed because it is an identifier rather than
a predictive feature.

---

## 3. Data Preparation

The data preparation process included:

- Data cleaning
- Data type correction
- Handling missing values
- Feature engineering
- Binary encoding
- Ordinal encoding
- One-hot encoding
- Train-test splitting

The final dataset contains **38 predictor variables**.

The data was divided using an **80/20 stratified train-test split**:

- Training set: **5,634 customers**
- Test set: **1,409 customers**

A fixed `random_state=42` was used to make the split reproducible.

---

## 4. Model Development

Three classification models were trained and compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

ROC-AUC was used as the primary comparison metric.

---

## 5. Model Performance

The model comparison results are available in:

`reports/model_comparison.csv`

The final model metrics are available in:

`reports/final_model_metrics.csv`

The selected model was:

**Logistic Regression**

Model selection was based on ROC-AUC performance.

---

## 6. Final Evaluation

The selected model was evaluated using:

- Classification report
- Confusion matrix
- ROC curve
- Precision-recall curve
- Threshold analysis
- Prediction error analysis

The detailed evaluation is documented in:

`notebooks/04_evaluation.ipynb`

---

## 7. Threshold Analysis

The project also examined different classification thresholds.

Changing the threshold changes the balance between:

- Identifying more potential churners
- Producing more false-positive predictions

This is important from a business perspective because the optimal threshold
depends on the cost of missing a potential churner versus the cost of
contacting a customer who would not have churned.

The threshold analysis is available in:

`reports/threshold_analysis.csv`

---

## 8. Business Interpretation

The model should be treated as a **customer-risk prioritization tool**, not
as a guarantee that a customer will churn.

Customers with higher predicted churn probability can be prioritized for
retention actions such as targeted communication, offers, or customer-service
interventions.

The appropriate intervention threshold should depend on the company's
available retention resources and the relative cost of false negatives and
false positives.

---

## 9. Limitations

This project uses historical customer data, so model performance may change
as customer behavior, pricing, products, and market conditions change.

The model identifies statistical patterns associated with churn; it does not
establish that a particular factor causes churn.

The model should therefore be monitored and periodically re-evaluated when
new customer data becomes available.

---

## 10. Conclusion

This project demonstrates an end-to-end customer churn machine learning
workflow:

**Exploration → Preprocessing → Model Training → Evaluation → Business Findings**

The resulting model provides a systematic approach for identifying
customers with higher churn risk and can support more targeted customer
retention strategies.