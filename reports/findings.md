# Telco Churn — Business Findings

**One-page summary · model: Logistic Regression · held-out test ROC-AUC 0.84**

---

## The question

*Who will cancel, why, and what should the business do about it?* A telecom carrier
loses recurring revenue on every cancellation, and reacquiring a customer costs far
more than retaining one. We built a model that flags likely churners early enough
for the retention team to act.

## What the data said

- **7,043 customers; 26.5% churned** — an imbalanced target. A do-nothing model that
  predicts *"nobody churns"* scores 73.5% accuracy while catching **zero** leavers,
  so accuracy is a misleading headline here.
- One data-quality bug: `TotalCharges` was stored as text with 11 blank values, all
  for brand-new customers (`tenure = 0`). These are legitimately **$0**, not missing,
  and were set to zero rather than dropped.

## The model

Three classifiers were compared with 5-fold cross-validation; **Logistic Regression
won on the metrics that matter** and is also the most interpretable — the ideal
outcome.

| Model | ROC-AUC | Recall (churn) | Accuracy |
|---|---|---|---|
| **Logistic Regression** ✅ | **0.846** | **0.797** | 0.749 |
| Gradient Boosting | 0.829 | 0.700 | 0.771 |
| Random Forest | 0.824 | 0.465 | **0.788** |

> **The key lesson in one row:** Random Forest has the *highest accuracy* yet the
> *worst recall* — it looks best but misses **more than half** the churners. We
> selected on ROC-AUC and recall, not accuracy.

**On the held-out test set**, Logistic Regression achieved **ROC-AUC 0.842** and
**recovered 296 of 374 real churners (79%)**, at the cost of 298 false alarms — a
deliberate trade: a missed churner is lost revenue, a false alarm is one cheap,
unnecessary retention call.

## Why customers leave (the drivers)

Consistent across model coefficients, SHAP values, and the exploratory analysis:

1. **Month-to-month contracts** — by far the strongest churn signal; two-year
   contracts are the strongest protector.
2. **Short tenure** — the first year is the danger zone; risk falls steadily as
   customers stay.
3. **Fiber-optic internet at a high price point** — the highest-risk paid service,
   a value-for-money / expectations signal.
4. **Electronic-check payment and lacking tech-support / online-security** — friction
   and low engagement correlate with leaving.

## What the business should do

| Lever | Action | Rationale |
|---|---|---|
| **Contracts** | Incentivise month-to-month → 1/2-year (loyalty discount, bundled perk) | Largest single driver; converts the riskiest segment |
| **Onboarding** | Concierge outreach in the first 3–6 months | Churn concentrates in early tenure |
| **Fiber value** | Review fiber pricing/support; proactively support new fiber users | Highest-risk paid service |
| **Engagement** | Bundle online-security/tech-support; nudge autopay off electronic check | Sticky services raise switching cost |

**Highest-leverage single play:** move at-risk **month-to-month** customers onto
longer contracts. Every stage of the analysis points at that one sentence.

## How to use the model

Score any customer with `src/predict.py` (or the Streamlit app). Output is a churn
probability, a **risk band** (High ≥ 0.60 / Medium ≥ 0.35 / Low), and a suggested
action — so the retention team gets a ranked worklist, not a raw number.

## Caveats

- Snapshot data — no seasonality or time-to-churn; this predicts *whether*, not *when*.
- Coefficients show association, not proven causation; treat the levers as
  well-supported hypotheses to A/B test, not guarantees.
- The 0.60 / 0.35 thresholds are business choices — tune them to the retention team's
  capacity and the value of a saved customer.
