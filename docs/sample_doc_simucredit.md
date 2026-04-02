# SimuCredit Decisioning Model
## Model Development and Initial Validation Report

| | |
|---|---|
| **Document ID** | SDM-2026-LR-01 |
| **Version** | 1.0 |
| **Date** | April 02, 2026 |
| **Prepared By** | Quantitative Modeling Group — Model Risk & Analytics |
| **Classification** | Internal — Model Risk Management |
| **Regulatory Framework** | SR 11-7 \| ECOA / Reg B \| OCC 2011-12 \| EEOC 4/5ths Rule |

---

## 1. Executive Summary

This report documents the development and initial validation of the **SimuCredit Decisioning Model (SDM-2026-LR-01)**, a binary classification model designed to support credit approval decisions at the point of application. The model outputs a probability of loan denial for each applicant; a decision threshold of **p = 0.50** converts this to a binary approve/deny outcome.

The model was developed under the institution's Model Risk Management Policy and the supervisory guidance set forth in **SR 11-7** (*Guidance on Model Risk Management*, Federal Reserve, 2011). Independent validation is required prior to any production deployment.

### 1.1 Performance Summary

| Metric | Train | Test | Status |
|---|---|---|---|
| AUC (ROC) | 0.8100 | 0.8110 | ✅ Acceptable (> 0.70) |
| Gini Coefficient | 0.6199 | 0.6220 | — |
| KS Statistic | — | 0.5046 | ✅ Acceptable (> 0.30) |
| Hosmer-Lemeshow p | — | 0.2410 | ✅ Acceptable fit (p > 0.05) |
| K-Fold CV AUC (k=5) | 0.8057 ± 0.0706 | — | ⚠️ Unstable |

### 1.2 Fair Lending Summary

| Protected Attribute | AIR (Approval Rate Ratio) | EEOC Threshold | Status |
|---|---|---|---|
| Gender (Female vs. Male) | 0.9382 | ≥ 0.80 | ✅ Pass |
| Race (Minority vs. Majority) | 0.7200 | ≥ 0.80 | ❌ FAIL — Adverse Impact Detected |

> ⚠️ **Critical Finding:** Race AIR falls below the EEOC 4/5ths threshold. Mitigation analysis is required before deployment. See Section 7.

---

## 2. Data Description

### 2.1 Dataset Overview

The dataset (*SimuCredit_v2*) contains **682 anonymized consumer credit applications** drawn from the institution's originations portfolio (2021–2023). Records were extracted from the loan origination system (LOS) and processed through the institution's standard data quality pipeline.

| Property | Value |
|---|---|
| Total Observations | 682 |
| Model Features | 7 (all numerical) |
| Protected Attributes | 2 (Gender, Race — excluded from model inputs) |
| Approved (Status = 0) | 528 (77.4%) |
| Denied (Status = 1) | 154 (22.6%) |
| Training Set | 545 observations (80%, stratified) |
| Test Set | 137 observations (20%, stratified) |
| Observation Period | 2021 Q1 – 2023 Q4 |
| Missing Values | < 3% across all features (imputed — see §4.2) |

### 2.2 Feature Definitions

| # | Feature | Type | Observed Range | Business Definition |
|---|---|---|---|---|
| 1 | Mortgage | Continuous (USD) | [0, 600,000] | Outstanding mortgage balance at time of application |
| 2 | Balance | Continuous (USD) | [0, 50,000] | Total revolving credit balance across all open accounts |
| 3 | Amount Past Due | Continuous (USD) | [0, 5,000] | Total amount currently past due across all accounts |
| 4 | Delinquency | Ordinal {0,1,2,3} | 0–3 | Maximum delinquency cycle in the past 24 months |
| 5 | Inquiry | Count | 0–5 | Hard credit inquiries in the past 12 months |
| 6 | Open Trade | Count | 0–14 | Number of currently open trade lines |
| 7 | Utilization | Ratio [0,1] | 0.00–1.00 | Aggregate credit utilization (balance / credit limit) |

### 2.3 Protected Attributes (Excluded from Model — Fairness Audit Only)

The following demographic fields are present in the dataset but are **explicitly excluded from all model training steps** per ECOA / Regulation B. They are retained in a separate dataframe used only for post-hoc adverse impact analysis.

| Attribute | Encoding | Protected Group | Reference Group | Proportion |
|---|---|---|---|---|
| Gender | Binary 0/1 | Female (1) | Male (0) | 45.2% Female / 54.8% Male |
| Race | Binary 0/1 | Minority (1) | Majority (0) | 28.0% Minority / 72.0% Majority |

### 2.4 Class Distribution and Imbalance

The dataset exhibits significant **class imbalance**, with denial events comprising only **22.6%** of all observations. This is consistent with the expected denial rate for a prime consumer credit portfolio.

```
  Status Distribution:
    Approved (0): ██████████████████████████████             528  (77.4%)
    Denied   (1): █████████                                  154  (22.6%)
    Total:        682
```

> **Note for Validation Team:** Class imbalance was handled via stratified splitting. No SMOTE or oversampling was applied. The default threshold of p=0.50 may be suboptimal — threshold sensitivity analysis is required during validation.

---

## 3. Regulatory and Governance Context

### 3.1 Applicable Regulatory Framework

| Regulation | Requirement |
|---|---|
| **SR 11-7** (Federal Reserve, 2011) | Primary model risk management framework. Mandates conceptual soundness review, ongoing monitoring, and independent validation for all models used in material decisions. |
| **ECOA / Regulation B** | Prohibits credit discrimination on the basis of race, color, religion, national origin, sex, marital status, or age. Adverse action notices required for all denied applications. |
| **OCC 2011-12** | Parallel OCC model risk guidance. Requires validation of all models used in credit decisioning, with documentation of limitations and assumptions. |
| **EEOC 4/5ths Rule** | Adverse Impact Ratio (AIR) ≥ 0.80 required for all protected groups. AIR = (Approval Rate of Protected Group) / (Approval Rate of Reference Group). |

### 3.2 Model Risk Tier Classification

This model is classified as **Tier 1 (High Risk)** under the Model Risk Policy due to its direct, automated impact on credit decisions affecting protected classes.

Tier 1 requirements:
- Full independent validation before production deployment
- Annual model performance review with documented findings
- Quarterly PSI-based monitoring reports
- Fair lending audit at each model update or threshold change
- Model Risk Committee (MRC) sign-off required for deployment

---

## 4. Model Development Methodology

### 4.1 Algorithm Selection Rationale

Logistic Regression was selected after evaluating four candidate algorithms:

| Algorithm | Train AUC | Test AUC | Interpretability | Selected |
|---|---|---|---|---|
| Logistic Regression | 0.8100 | 0.8110 | High (coefficient table) | ✅ Champion |
| Random Forest | 0.8400 | 0.8210 | Low (black box) | Challenger |
| XGBoost | 0.8500 | 0.8160 | Low | Rejected |
| Decision Tree (depth=4) | 0.7600 | 0.7510 | High | Rejected (lower AUC) |

Logistic Regression was selected based on:
1. **SR 11-7 interpretability requirement** — coefficients provide direct, auditable log-odds mapping
2. **Regulatory precedent** — industry-standard for consumer credit under ECOA
3. **Comparable performance** — AUC within acceptable range of the RF challenger

### 4.2 Preprocessing Pipeline

The following pipeline is applied identically to training data and all future scoring data. Parameters are fitted **exclusively on training data** and frozen before any test-set evaluation.

```
Pipeline:
  Step 1: Missing Value Imputation
    Method:  SimpleImputer(strategy='median')
    Applied: All 7 feature columns
    Fit on:  Training data only. Frozen for scoring.

  Step 2: Feature Scaling
    Method:  StandardScaler()  →  xʼ = (x − μ) / σ
    Applied: All 7 feature columns (post-imputation)
    Fit on:  Training data only. Frozen for scoring.

  Step 3: Protected Attribute Removal
    Gender and Race are dropped BEFORE model training.
    They are retained in a separate protected_data.csv for fairness audit.
```

### 4.3 Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    stratify     = y,       # preserves 6.5% denial rate in both splits
    random_state = 42
)
# Training: 545 rows  |  Test: 137 rows
```

### 4.4 Feature Selection — RCIT Independence Screening

A Randomized Conditional Independence Test (RCIT) was applied to screen features for conditional dependence on the target variable (Status). All seven features were retained (p < 0.05).

| Feature | RCIT p-value | Decision |
|---|---|---|
| Mortgage | 0.0030 | ✅ Retained (p < 0.05) |
| Balance | 0.0110 | ✅ Retained (p < 0.05) |
| Amount Past Due | 0.0004 | ✅ Retained (p < 0.05) |
| Delinquency | 0.0001 | ✅ Retained (p < 0.05) |
| Inquiry | 0.0280 | ✅ Retained (p < 0.05) |
| Open Trade | 0.0420 | ✅ Retained (p < 0.05) |
| Utilization | 0.0001 | ✅ Retained (p < 0.05) |

### 4.5 Model Specification — Logistic Regression

The model is specified as a binary logistic regression:

$$
\log\left(\frac{P(\text{Denial})}{1 - P(\text{Denial})}\right) = \beta_0 + \sum_{j=1}^{7} \beta_j \tilde{x}_j
$$

where $\tilde{x}_j$ denotes the standardized (zero-mean, unit-variance) version of feature $x_j$, and

$$
P(\text{Denial} \mid \mathbf{x}) = \sigma\!\left(\beta_0 + \boldsymbol{\beta}^\top \tilde{\mathbf{x}}\right) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^\top \tilde{\mathbf{x}})}}
$$

**Estimated Coefficients (on standardized features, L2 regularization, C=1.0):**

| Feature | β̂ (Std. Scale) | Approx. Std. Error | Odds Ratio (e^β̂) | Direction |
|---|---|---|---|---|
| Mortgage | +0.0373 | 0.0036 | 1.0380 | ↑ Higher → more likely denied |
| Balance | +0.2752 | 0.0312 | 1.3167 | ↑ Higher → more likely denied |
| Amount Past Due | +0.7340 | 0.0713 | 2.0834 | ↑ Higher → more likely denied |
| Delinquency | +0.8104 | 0.1025 | 2.2488 | ↑ Higher → more likely denied |
| Inquiry | +0.5871 | 0.0716 | 1.7988 | ↑ Higher → more likely denied |
| Open Trade | -0.0880 | 0.0114 | 0.9157 | ↓ Higher → less likely denied |
| Utilization | +0.4454 | 0.0770 | 1.5612 | ↑ Higher → more likely denied |
| Intercept (β₀) | -1.5319 | — | — | — |

**Interpretation example:** An applicant one standard deviation above the mean Utilization increases their log-odds of denial by +0.4454, corresponding to an odds multiplier of 1.561×.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Penalty | L2 (Ridge) | Prevents overfitting; standard for logistic regression |
| C (inverse regularization) | 1.0 | Default; no strong prior on coefficient magnitude |
| Solver | lbfgs | Efficient for small-to-medium datasets; supports L2 |
| Max iterations | 1,000 | Convergence confirmed (loss < 1e-4) |
| Class weight | None | No resampling; stratified split preserves class balance |
| Threshold | 0.50 | Default; sensitivity analysis recommended by validation team |

---

## 5. Model Performance Evaluation

### 5.1 Discriminatory Power

All metrics computed on the held-out test set (n=137) unless stated.

| Metric | Train Set | Test Set | Gap | Benchmark |
|---|---|---|---|---|
| AUC (area under ROC curve) | 0.8100 | 0.8110 | -0.0011 | > 0.70 = acceptable |
| Gini Coefficient (2·AUC−1) | 0.6199 | 0.6220 | -0.0021 | > 0.40 = acceptable |
| KS Statistic | — | 0.5046 | — | > 0.30 = acceptable |

> **Train/Test AUC gap** of -0.0011 (-0.1%) is within acceptable range (< 5% relative gap). Validation team should independently verify using k-fold CV.

### 5.2 Calibration Metrics

| Metric | Value | Interpretation |
|---|---|---|
| F1 Score | 0.5098 | Harmonic mean of precision and recall |
| Precision | 0.6500 | Of predicted denials, fraction that are true denials |
| Recall (Sensitivity) | 0.4194 | Of actual denials, fraction correctly flagged |
| Specificity | 0.9340 | Of actual approvals, fraction correctly approved |

### 5.3 Confusion Matrix (Test Set, threshold = 0.50)

```
                        PREDICTED
                   Approve        Deny
          ┌─────────────────────────────┐
  ACTUAL  │ Approve     99             7 │
          │ Deny        18            13 │
          └─────────────────────────────┘
  True Negative Rate (Specificity):  0.9340
  True Positive Rate (Sensitivity):  0.4194
```

### 5.4 Hosmer-Lemeshow Goodness-of-Fit Test

The Hosmer-Lemeshow test partitions applicants into deciles by predicted probability and compares observed vs. expected denial counts:

$$
\chi^2_{HL} = \sum_{g=1}^{G} \frac{(O_g - E_g)^2}{E_g(1 - E_g/n_g)}
$$

| Statistic | Value |
|---|---|
| χ² (Hosmer-Lemeshow) | 10.3548 |
| Degrees of Freedom | 8 (G − 2 = 10 − 2) |
| p-value | 0.2410 |
| H₀ | No significant difference between observed and expected frequencies |
| Decision | Fail to reject H₀ — model fit is acceptable |

### 5.5 K-Fold Cross-Validation (k=5, Stratified)

| Fold | AUC |
|---|---|
| Fold 1 | 0.8904 |
| Fold 2 | 0.7048 |
| Fold 3 | 0.7988 |
| Fold 4 | 0.7573 |
| Fold 5 | 0.8774 |
| **Mean** | **0.8057** |
| **Std Dev** | **0.0706** |
| **95% CI** | **[0.6675, 0.9440]** |

> Standard deviation of 0.0706 indicates fold-to-fold instability (σ ≥ 0.05) — investigate data heterogeneity.

---

## 6. Multicollinearity Analysis — Variance Inflation Factors

VIF is computed to detect problematic linear dependencies among predictors. High VIF inflates coefficient standard errors and reduces interpretability.

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

where $R_j^2$ is the coefficient of determination from regressing feature $j$ on all other features.

| Feature | VIF | Interpretation |
|---|---|---|
| Mortgage | 1.013 | ✅ No concern |
| Balance | 1.008 | ✅ No concern |
| Amount Past Due | 1.010 | ✅ No concern |
| Delinquency | 1.018 | ✅ No concern |
| Inquiry | 1.004 | ✅ No concern |
| Open Trade | 1.017 | ✅ No concern |
| Utilization | 1.004 | ✅ No concern |

> All VIF values are within acceptable bounds (< 5). No multicollinearity remediation required. Coefficient estimates are stable.

---

## 7. Fair Lending Analysis — Adverse Impact Assessment

### 7.1 Methodology

Adverse Impact Ratio (AIR) is computed per the EEOC Uniform Guidelines on Employee Selection Procedures (29 C.F.R. § 1607), adapted to credit decisioning per Interagency Fair Lending Examination Procedures (CFPB / OCC / FDIC, 2009):

$$
\text{AIR} = \frac{P(\text{Approved} \mid \text{Protected Group})}{P(\text{Approved} \mid \text{Reference Group})}
$$

A model has **adverse impact** if AIR < 0.80 for any protected group (the EEOC 4/5ths, or 80%, rule).

### 7.2 Adverse Impact Results

| Protected Attribute | Protected Group | Reference Group | Approval Rate (Protected) | Approval Rate (Reference) | AIR | Pass? |
|---|---|---|---|---|---|---|
| Gender | Female | Male | 0.746 | 0.795 | 0.9382 | ✅ Pass |
| Race | Minority | Majority | 0.600 | 0.833 | 0.7200 | ❌ FAIL |

### 7.3 Findings and Required Actions

**Gender:** AIR = 0.9382 ≥ 0.80. No adverse impact at the current threshold. Continued monitoring required.

**Race:** AIR = 0.7200 < 0.80. **Adverse impact detected.** The following remediation steps are required:
  1. Slicing fairness analysis to identify which feature segments drive the disparity
  2. Threshold adjustment analysis — determine whether adjusting the decision threshold improves AIR without unacceptable accuracy loss
  3. Feature binning mitigation — evaluate whether binning proxy features reduces disparate impact
  4. If mitigation is insufficient: escalate to Model Risk Committee and Legal for pre-deployment regulatory review

> **Validation Requirement:** The independent validation team must re-run this analysis on the rebuilt model, perform slicing fairness analysis by feature segment, and document all mitigation steps taken.

---

## 8. Known Limitations and Model Assumptions

| # | Limitation | Risk | Required Mitigation |
|---|---|---|---|
| 1 | Class imbalance (denial rate 22.6%) may bias toward approval; threshold sensitivity untested | Medium | Threshold sensitivity analysis during validation |
| 2 | No temporal ordering in train/test split; model may not capture economic cycle effects | Medium | PSI-based drift monitoring, quarterly |
| 3 | Linear log-odds assumption of logistic regression may miss nonlinear effects | Low | Validated via HL test and residual analysis; Random Forest challenger provides nonlinear benchmark |
| 4 | Protected attribute proxy risk: Utilization and Balance may be correlated with race/gender | High | Slicing fairness analysis; evaluate fairness-aware preprocessing for future model version |
| 5 | Model trained on single institution portfolio; generalizability to other portfolios is unvalidated | Low | Out-of-scope for current validation; flag for any portfolio expansion |

---

## 9. Model Artifacts and Reproducibility

### 9.1 Artifact Inventory

| Artifact | Filename | Format | Location |
|---|---|---|---|
| This report | sample_doc_simucredit.md | Markdown | /docs/ |

### 9.2 Environment and Dependencies

```
python        3.11.4
scikit-learn  1.3.2
pandas        2.1.4
numpy         1.26.2
scipy         1.11.4
modeva        0.9.1
```

### 9.3 Reproduction Instructions

The validation team should reconstruct the model independently from this specification:

```python
# 1. Load data
import pandas as pd
df = pd.read_csv('SimuCredit_v2.csv')

# 2. Define features and target
FEATURES = ['Mortgage','Balance','Amount Past Due',
            'Delinquency','Inquiry','Open Trade','Utilization']
X = df[FEATURES]; y = df['Status']

# 3. Split (must match: test_size=0.20, stratify=y, random_state=42)
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# 4. Preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='median').fit(X_tr)
scl = StandardScaler().fit(imp.transform(X_tr))
X_tr_s = scl.transform(imp.transform(X_tr))
X_te_s = scl.transform(imp.transform(X_te))

# 5. Train
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_tr_s, y_tr)

# 6. Evaluate — expected: AUC ≈ 0.8110, KS ≈ 0.5046
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_te, model.predict_proba(X_te_s)[:,1])
```

---

## 10. Validation Checklist and Sign-off

The following items are submitted to the independent validation team for review:

| # | Validation Item | Status |
|---|---|---|
| 1 | Independent model reconstruction from this specification | ⬜ Pending |
| 2 | Performance validation (AUC, KS, Gini, HL, K-Fold CV) | ⬜ Pending |
| 3 | Robustness testing (feature perturbation, resilience) | ⬜ Pending |
| 4 | Fairness audit — AIR for all protected attributes | ⬜ Pending |
| 5 | Fairness mitigation analysis (if AIR < 0.80) | ⬜ Pending |
| 6 | Interpretability review (coefficients, SHAP, PDP) | ⬜ Pending |
| 7 | Challenger model comparison (LR vs. RF) | ⬜ Pending |
| 8 | Validation report issuance | ⬜ Pending |
| 9 | Model Risk Committee (MRC) review and approval | ⬜ Pending |

| Role | Name | Signature | Date |
|---|---|---|---|
| Model Developer | [Redacted] | __________ | April 02, 2026 |
| Model Owner (Business Line) | [Redacted] | __________ | April 02, 2026 |
| Independent Validation Lead | *Pending* | __________ | — |
| Chief Risk Officer | *Pending* | __________ | — |
| MRC Chair | *Pending* | __________ | — |
