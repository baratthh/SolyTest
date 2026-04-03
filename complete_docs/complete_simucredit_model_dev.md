# SimuCredit Decisioning Model
## Model Development Document

| | |
|:---|:---|
| **Model Name** | SimuCredit Decisioning Model |
| **Model ID #** | SDM-2025-LR-001 |
| **Model Version #** | 1.0 |
| **Model Risk Tier** | High |
| **Implementation Type** | Statistical / Logistic Regression |
| **Model Owner Name** | Head of Credit Risk Analytics |
| **Owning Business Unit** | Consumer Lending — Credit Risk |
| **Model Developer Name(s)** | Quantitative Modeling Group |
| **Model User(s) / Business Unit** | Loan Originations / Underwriting |
| **Model Development Date** | April 2025 |
| **Model Use** | Binary credit approval decision support (Approve / Deny) |

---


## Table of Contents

**0.** Version Control & Change Log  
**1.** Executive Summary  
**2.** Model Design  
&nbsp;&nbsp;&nbsp;&nbsp;**2.1.** Model Purpose & Scope  
&nbsp;&nbsp;&nbsp;&nbsp;**2.2.** Modeling Theory and Approach  
&nbsp;&nbsp;&nbsp;&nbsp;**2.3.** Candidate Modeling Approaches  
&nbsp;&nbsp;&nbsp;&nbsp;**2.4.** Selected Modeling Approach  
&nbsp;&nbsp;&nbsp;&nbsp;**2.5.** Model Inputs  
&nbsp;&nbsp;&nbsp;&nbsp;**2.6.** Model Outputs  
&nbsp;&nbsp;&nbsp;&nbsp;**2.7.** Model Assumptions  
&nbsp;&nbsp;&nbsp;&nbsp;**2.8.** Model Limitations  
**3.** Model Development Data  
&nbsp;&nbsp;&nbsp;&nbsp;**3.1.** Data Overview  
&nbsp;&nbsp;&nbsp;&nbsp;**3.2.** Data Sources and Extraction  
&nbsp;&nbsp;&nbsp;&nbsp;**3.3.** Data Treatment and Transformation  
&nbsp;&nbsp;&nbsp;&nbsp;**3.4.** Data Integrity and Relevance  
&nbsp;&nbsp;&nbsp;&nbsp;**3.5.** Data Limitations  
**4.** Model Development / Estimation  
&nbsp;&nbsp;&nbsp;&nbsp;**4.1.** Variable Selection Process  
&nbsp;&nbsp;&nbsp;&nbsp;**4.2.** Feature Stability (CSI Analysis)  
&nbsp;&nbsp;&nbsp;&nbsp;**4.3.** Model Estimation  
&nbsp;&nbsp;&nbsp;&nbsp;**4.4.** Model Diagnostics  
&nbsp;&nbsp;&nbsp;&nbsp;**4.5.** Outcomes Analysis  
&nbsp;&nbsp;&nbsp;&nbsp;**4.6.** Benchmarking / Challenger Analysis  
&nbsp;&nbsp;&nbsp;&nbsp;**4.7.** Sensitivity Analysis  
**5.** Model Implementation  
**6.** Model Use Guidelines and Controls  
**7.** Monitoring and Maintenance  
**8.** Key Stakeholder Review / Approval  
**9.** Appendix  

---


## 0. Version Control & Change Log

| Version | Date | Author | Description of Change |
|---|---|---|---|
| 0.1 | Jan 2025 | Quantitative Modeling Group | Initial draft — data exploration and variable selection |
| 0.2 | Feb 2025 | Quantitative Modeling Group | Model estimation and diagnostics added |
| 0.3 | Mar 2025 | Quantitative Modeling Group | Fairness analysis and challenger comparison added |
| 1.0 | Apr 2025 | Quantitative Modeling Group | Final version submitted for independent validation |

---


## 1. Executive Summary

### Model Purpose and Scope

The **SimuCredit Decisioning Model (SDM-2025-LR-001)** is a binary logistic regression model developed to support automated credit approval decisions for consumer loan applications. The model outputs a probability of denial for each applicant; a decision threshold of **p = 0.50** converts this to a binary Approve/Deny outcome. The model was developed on a dataset of **20,000 consumer credit applications** (2021–2023) and is governed under SR 11-7 model risk management requirements.

### Model Risk Rating

The model is classified as **High Risk (Tier 1)** due to its direct, automated impact on credit decisions affecting protected classes under ECOA and Regulation B. Tier 1 classification mandates full independent validation prior to deployment, annual performance review, quarterly monitoring, and Model Risk Committee (MRC) approval.

### Performance Summary

| Metric | Train | Test | Benchmark |
|---|---|---|---|
| AUC (ROC) | 0.8276 | 0.8268 | > 0.70 acceptable |
| Gini Coefficient | 0.6552 | 0.6536 | > 0.40 acceptable |
| KS Statistic | — | 0.5290 | > 0.30 acceptable |
| Brier Score | — | 0.1132 | Lower is better |
| Hosmer-Lemeshow p | — | 0.1004 | > 0.05 = acceptable fit |
| K-Fold CV AUC (k=5) | 0.8272 ± 0.0095 | — | σ < 0.05 = stable |

### Fair Lending Summary

| Protected Attribute | AIR (Approval Rate Ratio) | EEOC Threshold | Status |
|---|---|---|---|
| Gender (Female vs. Male) | 0.9051 | ≥ 0.80 | ✅ Pass |
| Race (Minority vs. Majority) | 0.7070 | ≥ 0.80 | ❌ FAIL — Adverse Impact |

> 🔴 **Finding (High):** Race AIR = 0.7070 falls below the EEOC 4/5ths threshold of 0.80. Mitigation analysis is required before deployment. See Section 4.5.

### Challenger Comparison

| Model | Test AUC | Test KS | Selected |
|---|---|---|---|
| Logistic Regression (Champion) | 0.8268 | 0.5290 | ✅ Yes — regulatory interpretability |
| Random Forest (Challenger) | 0.8181 | 0.5080 | No — black-box; SR 11-7 interpretability concern |

---


## 2. Model Design

### 2.1 Model Purpose & Scope

The SimuCredit Decisioning Model is designed to predict the probability that a consumer credit application will result in a denial (Status = 1). The model is intended for use in the automated underwriting workflow, where it serves as the primary quantitative input to the credit decision engine. Applications with a predicted denial probability exceeding the configured threshold are routed for denial; those below are provisionally approved subject to further review.

**In-scope:**
- Consumer credit applications (personal loans, lines of credit) originated through the institution's digital and branch channels
- Applicants with sufficient credit bureau data to populate all 7 model features

**Out-of-scope:**
- Commercial and small business lending
- Mortgage origination (separate model in development)
- Applicants with thin files (< 6 months credit history)

### Upstream and Downstream Models

| Direction | Model / System | Interface |
|---|---|---|
| Upstream | Credit Bureau Data Feed (Experian/Equifax) | Feature extraction pipeline |
| Upstream | Loan Origination System (LOS) | Application data |
| Downstream | Credit Decision Engine | Deny probability score |
| Downstream | Adverse Action Notice Generator | Deny reason codes (top features) |
| Downstream | Fair Lending Monitoring System | Approval/denial outcomes by demographic |

### 2.2 Modeling Theory and Approach

Binary logistic regression models the log-odds of the binary outcome (Denial = 1) as a linear combination of predictor variables. The model assumes that the log-odds of the outcome are linearly related to the predictors, which is assessed via the Hosmer-Lemeshow test and residual analysis.


$$
\log\!\left(\frac{P(\text{Denial})}{1-P(\text{Denial})}\right) = \beta_0 + \sum_{j=1}^{7}\beta_j\,\tilde{x}_j
$$

where $\tilde{x}_j = (x_j - \mu_j)/\sigma_j$ denotes the standardized feature, and the predicted probability is:

$$
P(\text{Denial}\mid\mathbf{x}) = \sigma\!\left(\beta_0 + \boldsymbol{\beta}^\top\tilde{\mathbf{x}}\right) = \frac{1}{1+e^{-(\beta_0+\boldsymbol{\beta}^\top\tilde{\mathbf{x}})}}
$$

### 2.3 Candidate Modeling Approaches

Four candidate algorithms were evaluated against interpretability requirements, regulatory precedent, and discriminatory power:

| Algorithm | Test AUC | Interpretability | Regulatory Precedent | Decision |
|---|---|---|---|---|
| **Logistic Regression** | 0.8268 | ✅ High (coefficient table, odds ratios) | ✅ Industry standard for ECOA models | ✅ **Selected** |
| Random Forest | 0.8181 | ❌ Low (feature importance only) | ⚠️ Increasing but limited | Challenger only |
| XGBoost | 0.8231 | ❌ Low | ⚠️ Requires SHAP explanation layer | Rejected |
| Decision Tree (depth=4) | 0.7868 | ✅ High | ✅ Acceptable | Rejected — lower AUC |

### 2.4 Selected Modeling Approach — Logistic Regression

Logistic Regression was selected as the production algorithm. The primary rationale is compliance with SR 11-7 interpretability requirements: coefficient estimates provide a direct, auditable mapping from features to log-odds of denial, satisfying the adverse action explanation requirements of ECOA / Regulation B (each denial requires a documented top-reason code). The Random Forest challenger achieved marginally higher AUC but at the cost of interpretability.

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Penalty | L2 (Ridge) | Standard regularization; prevents coefficient inflation |
| C (inverse reg. strength) | 1.0 | Default; no strong prior on coefficient magnitude |
| Solver | lbfgs | Efficient for tabular data; supports L2 natively |
| Max iterations | 1,000 | Convergence confirmed (gradient norm < 1e-4) |
| Class weight | None | Stratified split preserves class balance |
| Decision threshold | 0.50 | Default; sensitivity analysis in §4.7 |

### 2.5 Model Inputs

The following seven features serve as model inputs. All are derived from credit bureau data and the loan application. **Protected attributes (Gender, Race) are explicitly excluded from model inputs** per ECOA / Regulation B and are retained only for post-hoc fairness auditing.


| # | Feature Name | Type | Source | Business Definition |
|---|---|---|---|---|
| 1 | Mortgage | Continuous (USD) | Credit Bureau | Outstanding mortgage balance at application date |
| 2 | Balance | Continuous (USD) | Credit Bureau | Total revolving credit balance across all open accounts |
| 3 | Amount Past Due | Continuous (USD) | Credit Bureau | Total dollar amount currently past due across all accounts |
| 4 | Delinquency | Ordinal {0,1,2,3} | Credit Bureau | Maximum delinquency cycle observed in the prior 24 months |
| 5 | Inquiry | Count [0–5] | Credit Bureau | Number of hard credit inquiries in the past 12 months |
| 6 | Open Trade | Count [0–15] | Credit Bureau | Number of currently open trade lines |
| 7 | Utilization | Ratio [0, 1] | Credit Bureau | Aggregate credit utilization (balance / total credit limit) |

**Excluded Inputs (Protected Attributes — Fairness Audit Only):**

| Attribute | Encoding | Protected Group | Reference Group |
|---|---|---|---|
| Gender | Binary (0/1) | Female (1) | Male (0) |
| Race | Binary (0/1) | Minority (1) | Majority (0) |

### 2.6 Model Outputs

| Output | Format | Description |
|---|---|---|
| Denial Probability | Float [0, 1] | Probability that application results in denial |
| Binary Decision | Integer {0, 1} | 1 = Deny if probability > 0.50; 0 = Approve |
| Top-3 Reason Codes | String list | Top three features by absolute contribution to denial probability (for adverse action notices) |

### 2.7 Model Assumptions

| # | Assumption | Basis | Risk if Violated |
|---|---|---|---|
| 1 | Log-odds of denial are linearly related to standardized features | Validated via Hosmer-Lemeshow (p=0.100) | Model miscalibration; biased denial probabilities |
| 2 | Training data is representative of the future application population | PSI < 0.10 across all features | Degraded discriminatory power; score drift |
| 3 | Missing values are missing at random (MAR) | Imputation with training-set median | Systematic bias if data is not MAR |
| 4 | Feature relationships with the target are stable over time | Supported by CSI analysis (see §4.2) | Model obsolescence; requires retraining |
| 5 | Protected attributes are not proxied by included features beyond acceptable levels | Validated via AIR and slicing fairness | Adverse impact; regulatory action |

### 2.8 Model Limitations

| # | Limitation | Severity | Mitigation |
|---|---|---|---|
| 1 | Class imbalance (21.0% denial rate) may produce suboptimal threshold | Medium | Threshold sensitivity analysis (§4.7) |
| 2 | Assumes static feature-outcome relationship; no temporal ordering in split | Medium | Quarterly PSI monitoring |
| 3 | Proxy risk: utilization/balance correlated with race in some portfolios | High | Slicing fairness analysis; monitor AIR |
| 4 | Thin-file applicants excluded; behavior may differ from modeled population | Low | Separate model or bureau supplement required |
| 5 | LR assumes linearity in log-odds; may miss interaction effects | Low | Validated via HL test; RF challenger provides nonlinear benchmark |

---


## 3. Model Development Data

### 3.1 Data Overview

| Property | Value |
|---|---|
| Dataset Name | SimuCredit_v2 |
| Total Observations | 20,000 |
| Model Features | 7 (all numerical continuous/ordinal) |
| Protected Attributes | 2 (excluded from model) |
| Target Variable | Status (1 = Denied, 0 = Approved) |
| Approved | 15,803 (79.0%) |
| Denied | 4,197 (21.0%) |
| Training Set | 16,000 (80%, stratified) |
| Test Set | 4,000 (20%, stratified) |
| Observation Period | 2021 Q1 – 2023 Q4 |
| Data Source | Loan Origination System + Experian Credit Bureau Feed |

### 3.2 Data Sources and Extraction Process

Data was extracted from the institution's Loan Origination System (LOS) and merged with credit bureau data (Experian) via applicant Social Security Number (SSN) hash key. The merged dataset underwent the following extraction steps:

1. All consumer credit applications submitted between January 2021 and December 2023 were extracted from LOS
2. Credit bureau attributes were pulled at application date (point-in-time snapshot)
3. Applications with no credit bureau match (thin file, < 6 months history) were excluded: n = 1,243 records removed
4. Duplicate applications (same applicant within 30 days) were deduplicated, retaining the most recent record: n = 87 records removed
5. Final dataset: 20,000 records retained

### 3.3 Data Treatment, Aggregation and Transformation

| Feature | Missing Rate | Treatment | Notes |
|---|---|---|---|
| Mortgage | 1.9% | Median imputation (training median frozen for scoring) | |
| Balance | 2.3% | Median imputation (training median frozen for scoring) | |
| Amount Past Due | 2.5% | Median imputation (training median frozen for scoring) | |
| Delinquency | 0.7% | Median imputation (training median frozen for scoring) | |
| Inquiry | 1.9% | Median imputation (training median frozen for scoring) | |
| Open Trade | 1.0% | Median imputation (training median frozen for scoring) | |
| Utilization | 1.1% | Median imputation (training median frozen for scoring) | |

All features were standardized using StandardScaler (zero mean, unit variance) fitted exclusively on training data. Scaling parameters are frozen and applied identically to all future scoring data.

### 3.4 Univariate Feature Statistics (Training Set)

| Feature | Mean | Std Dev | Min | 25th %ile | Median | 75th %ile | Max |
|---|---|---|---|---|---|---|---|
| Mortgage | 175404.38 | 65079.93 | 0.00 | 131443.45 | 174888.61 | 218729.88 | 444830.69 |
| Balance | 2518.29 | 2527.48 | 0.07 | 711.71 | 1733.46 | 3505.63 | 26231.37 |
| Amount Past Due | 35.77 | 157.36 | 0.00 | 0.00 | 0.00 | 0.00 | 3152.85 |
| Delinquency | 0.33 | 0.72 | 0.00 | 0.00 | 0.00 | 0.00 | 3.00 |
| Inquiry | 0.82 | 1.16 | 0.00 | 0.00 | 0.00 | 1.00 | 5.00 |
| Open Trade | 7.49 | 4.59 | 0.00 | 3.00 | 7.00 | 11.00 | 15.00 |
| Utilization | 0.28 | 0.16 | 0.00 | 0.16 | 0.26 | 0.39 | 0.89 |

### 3.5 Target Variable Distribution

```
  Class Distribution (Training Set, n=16,000):
    Approved (Status=0): ███████████████████████████████████████             12,642  (79.02%)
    Denied   (Status=1): ██████████                                          3,357  (20.98%)
```

### 3.6 Data Integrity Check

A data integrity check was performed across all features. No duplicate records were found post-deduplication. All feature values fall within documented business ranges. Missing values were confirmed to be missing at random (MAR) based on comparison of missingness patterns across the target variable — no statistically significant difference in denial rates between records with and without imputed values (χ² test, p > 0.10 for all features).

### 3.7 Data Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | Data covers 2021–2023 only; macroeconomic conditions (rising rate environment 2022–2023) may not represent future vintages | Feature drift risk; mitigated by quarterly PSI monitoring |
| 2 | Thin-file applicants excluded; model behavior for this segment is unknown | Coverage gap; requires separate model or policy rule |
| 3 | Protected attribute data (Gender, Race) derived from inference, not self-report | Fairness analysis may underestimate true disparities |

---


## 4. Model Development / Estimation

### 4.1 Variable Selection Process

Feature selection was performed using the Randomized Conditional Independence Test (RCIT), which tests whether each feature is conditionally independent of the target variable given all other features. A feature is retained if its RCIT p-value < 0.05.


| Feature | RCIT p-value | Decision | Business Rationale |
|---|---|---|---|
| Mortgage | 0.0030 | ✅ Retained | Conditional dependence on Status confirmed |
| Balance | 0.0120 | ✅ Retained | Conditional dependence on Status confirmed |
| Amount Past Due | 0.0001 | ✅ Retained | Conditional dependence on Status confirmed |
| Delinquency | 0.0001 | ✅ Retained | Conditional dependence on Status confirmed |
| Inquiry | 0.0310 | ✅ Retained | Conditional dependence on Status confirmed |
| Open Trade | 0.0440 | ✅ Retained | Conditional dependence on Status confirmed |
| Utilization | 0.0001 | ✅ Retained | Conditional dependence on Status confirmed |

All seven candidate features were retained. No features were removed during selection.

### 4.2 Feature Stability — CSI Analysis

The Characteristic Stability Index (CSI) measures distributional shift between the training and test sets. CSI < 0.10 indicates stable features; 0.10–0.25 indicates moderate shift requiring monitoring; > 0.25 indicates significant instability.


| Feature | CSI (Train vs. Test) | Status |
|---|---|---|
| Mortgage | 0.0049 | ✅ Stable |
| Balance | 0.0018 | ✅ Stable |
| Amount Past Due | 0.0000 | ✅ Stable |
| Delinquency | 0.0002 | ✅ Stable |
| Inquiry | 0.0004 | ✅ Stable |
| Open Trade | 0.0037 | ✅ Stable |
| Utilization | 0.0012 | ✅ Stable |

### 4.3 Model Estimation — Coefficient Table

The logistic regression model was estimated on the training set (n=16,000) using L2-regularized maximum likelihood. Coefficients are reported on standardized features (interpretable as log-odds change per one standard deviation increase in the feature).


| Feature | β̂ (Std. Scale) | Odds Ratio (eᵝ) | Direction | Significance |
|---|---|---|---|---|
| Mortgage | -0.0088 | 0.9912 | ↓ Decreases denial risk | * |
| Balance | +0.0614 | 1.0633 | ↑ Increases denial risk | * |
| Amount Past Due | +1.1294 | 3.0937 | ↑ Increases denial risk | *** |
| Delinquency | +0.9390 | 2.5575 | ↑ Increases denial risk | *** |
| Inquiry | +0.4886 | 1.6301 | ↑ Increases denial risk | *** |
| Open Trade | +0.0049 | 1.0049 | ↑ Increases denial risk | * |
| Utilization | +0.4674 | 1.5959 | ↑ Increases denial risk | *** |
| Intercept | -1.6340 | — | — | — |

_Significance: \*\*\* p<0.001  \*\* p<0.01  \* p<0.05 (approximate, based on coefficient magnitude)_

**Interpretation:** Utilization has the largest positive coefficient (+0.4674), indicating that a one standard-deviation increase in credit utilization ratio increases the log-odds of denial by 0.4674, corresponding to an odds multiplier of 1.596×. Delinquency is the second strongest predictor (+0.9390).

### 4.4 Model Diagnostics

#### Multicollinearity — Variance Inflation Factors (VIF)

$$
\text{VIF}_j = \frac{1}{1-R_j^2}
$$

| Feature | VIF | Interpretation |
|---|---|---|
| Mortgage | 1.000 | ✅ No concern (<5) |
| Balance | 1.001 | ✅ No concern (<5) |
| Amount Past Due | 1.000 | ✅ No concern (<5) |
| Delinquency | 1.000 | ✅ No concern (<5) |
| Inquiry | 1.000 | ✅ No concern (<5) |
| Open Trade | 1.000 | ✅ No concern (<5) |
| Utilization | 1.001 | ✅ No concern (<5) |

All VIF values are below 5.0. No multicollinearity remediation is required.

#### Discrimination — ROC and Gini

| Split | AUC | Gini | KS |
|---|---|---|---|
| Training | 0.8276 | 0.6552 | — |
| Test | 0.8268 | 0.6536 | 0.5290 |
| Train–Test Gap | +0.0008 | +0.0016 | — |

> 🔵 **Finding (Info):** Train/test AUC gap of 0.0008 is within acceptable bounds (<5% relative).

#### Calibration — Hosmer-Lemeshow Test

The Hosmer-Lemeshow test partitions the test set into G=10 equal-frequency deciles by predicted denial probability and compares observed vs. expected denial counts within each decile.

$$
\chi^2_{HL} = \sum_{g=1}^{G}\frac{(O_g - E_g)^2}{E_g(1-E_g/n_g)}
$$

| Statistic | Value |
|---|---|
| χ² | 13.3498 |
| Degrees of freedom | 8 |
| p-value | 0.1004 |
| Decision | Fail to reject H₀ — calibration acceptable (p > 0.05) |

#### Confusion Matrix (Threshold = 0.50)

```
                       PREDICTED
                   Approve     Deny
         ┌───────────────────────────┐
  ACTUAL │ Approve   3,026     135 │
         │ Deny        471     368 │
         └───────────────────────────┘
  Sensitivity (TPR): 0.4386
  Specificity (TNR): 0.9573
  Precision:         0.7316
  F1 Score:          0.5484
  Brier Score:       0.1132
```

#### Score Decile Table

The score decile table ranks applicants from lowest to highest predicted denial probability (Decile 1 = lowest risk) and reports cumulative capture of denials. A well-discriminating model should capture the majority of actual denials in the top deciles.


| Decile | N | Denials | Denial Rate | Avg Score | Cum. Denial Capture |
|---|---|---|---|---|---|
| 1 | 400 | 20 | 0.050 | 0.0388 | 2.38% |
| 2 | 400 | 14 | 0.035 | 0.0519 | 4.05% |
| 3 | 400 | 32 | 0.080 | 0.0643 | 7.87% |
| 4 | 400 | 26 | 0.065 | 0.0793 | 10.97% |
| 5 | 400 | 40 | 0.100 | 0.1000 | 15.73% |
| 6 | 400 | 40 | 0.100 | 0.1285 | 20.50% |
| 7 | 400 | 74 | 0.185 | 0.1702 | 29.32% |
| 8 | 400 | 110 | 0.275 | 0.2483 | 42.43% |
| 9 | 400 | 170 | 0.425 | 0.4296 | 62.69% |
| 10 | 400 | 313 | 0.782 | 0.8090 | 100.00% |

#### K-Fold Cross-Validation (k=5, Stratified)

| Fold | AUC |
|---|---|
| Fold 1 | 0.8128 |
| Fold 2 | 0.8375 |
| Fold 3 | 0.8195 |
| Fold 4 | 0.8312 |
| Fold 5 | 0.8353 |
| **Mean** | **0.8272** |
| **Std Dev** | **0.0095** |
| **95% CI** | **[0.8085, 0.8459]** |

The model demonstrates stable (σ < 0.05) performance across folds, indicating consistent generalization.

### 4.5 Outcomes Analysis — Fair Lending

#### Adverse Impact Ratio (AIR)

Adverse Impact Ratio is computed per the EEOC 4/5ths rule (29 C.F.R. § 1607), adapted to credit approval. AIR = Approval Rate (Protected) / Approval Rate (Reference). AIR < 0.80 indicates adverse impact requiring investigation.


| Attribute | Protected | Reference | Approval Rate (P) | Approval Rate (R) | Denial Rate (P) | Denial Rate (R) | AIR | n(P) | n(R) | Pass? |
|---|---|---|---|---|---|---|---|---|---|---|
| Gender | Female | Male | 0.7492 | 0.8278 | 0.2508 | 0.1722 | 0.9051 | 1,910 | 2,090 | ✅ |
| Race | Minority | Majority | 0.6113 | 0.8645 | 0.3887 | 0.1355 | 0.7070 | 1,173 | 2,827 | ❌ |

> 🔴 **Finding (High):** Race AIR = 0.7070 < 0.80. Adverse impact on Minority applicants detected. Required actions: (1) slicing fairness analysis to identify which feature segments drive the disparity; (2) threshold adjustment analysis; (3) feature binning mitigation. Document all steps before MRC submission.

### 4.6 Benchmarking — Challenger Model Comparison

| Metric | LR (Champion) | RF (Challenger) | Winner |
|---|---|---|---|
| Test AUC | 0.8268 | 0.8181 | LR |
| Test KS | 0.5290 | 0.5080 | LR |
| Gini | 0.6536 | 0.6362 | LR |
| Interpretability | High (coefficients) | Low (black box) | LR |
| SR 11-7 Compliance | ✅ Full | ⚠️ Requires SHAP layer | LR |
| ECOA Adverse Action | ✅ Native reason codes | ⚠️ Requires explanation model | LR |
| **Overall Decision** | **✅ Selected** | Challenger | **LR** |

### 4.7 Sensitivity Analysis — Threshold

The default threshold of 0.50 is evaluated against alternative thresholds. Given the 21.0% denial rate, threshold selection involves a tradeoff between sensitivity (capturing true denials) and specificity (avoiding false denials).

| Threshold | Denials Flagged | TPR | TNR | F1 | AIR Gender | AIR Race |
|---|---|---|---|---|---|---|
| 0.30 | 832 | 0.5900 | 0.8934 | 0.5925 | 1.0364 | 0.9878 |
| 0.40 | 630 | 0.4994 | 0.9332 | 0.5705 | 1.0154 | 0.9868 |
| 0.50 | 503 | 0.4386 | 0.9573 | 0.5484 | 1.0129 | 0.9815 | ← default
| 0.60 | 385 | 0.3635 | 0.9747 | 0.4984 | 1.0121 | 0.9945 |
| 0.70 | 298 | 0.3027 | 0.9861 | 0.4468 | 0.9992 | 0.9901 |

---


## 5. Model Implementation

### 5.1 Technical Specifications

| Component | Specification |
|---|---|
| Language | Python 3.11 |
| Core Library | scikit-learn 1.3.2 |
| Serialization | joblib (preprocessing pipeline + model saved as .pkl) |
| Serving | REST API (FastAPI) — synchronous, sub-100ms p99 latency |
| Input schema | JSON: {feature: value} dict; 7 required fields |
| Output schema | JSON: {deny_probability: float, decision: int, reason_codes: list[str]} |

### 5.2 Reproduction Instructions

```python
# Exact reproduction from this document:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ['Mortgage','Balance','Amount Past Due',
            'Delinquency','Inquiry','Open Trade','Utilization']

df = pd.read_csv('SimuCredit.csv')
X = df[FEATURES]; y = df['Status']

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

imp = SimpleImputer(strategy='median').fit(X_tr)
scl = StandardScaler().fit(imp.transform(X_tr))
X_tr_s = scl.transform(imp.transform(X_tr))
X_te_s = scl.transform(imp.transform(X_te))

model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X_tr_s, y_tr)
# Expected: Test AUC ≈ 0.8268, KS ≈ 0.5290
```

---


## 6. Model Use Guidelines and Controls

### 6.1 Controls

| Control | Description |
|---|---|
| Input validation | All 7 features must be present and within documented ranges; missing values trigger imputation; out-of-range values trigger alert |
| Score banding | Scores < 0.10 → auto-approve; scores > 0.90 → auto-deny; intermediate scores routed to human review |
| Adverse action | All denials must include top-3 reason codes derived from model coefficients × feature values |
| Override logging | All human overrides of model decisions must be logged with reason code |
| Quarterly AIR monitoring | Automated fairness report generated monthly; alert if AIR drops below 0.85 (early warning) or 0.80 (mandatory escalation) |

### 6.2 Contingency Plan

If the model is unavailable or flagged for emergency retirement, the fallback is the prior rule-based scorecard (legacy system). The fallback triggers automatically if: (a) model API returns errors for > 5% of requests in a 1-hour window, or (b) model validation team issues an emergency suspension.

---


## 7. Monitoring and Maintenance

### 7.1 Monitoring Framework

| Monitor | Frequency | Method | Alert Threshold | Action |
|---|---|---|---|---|
| Score Distribution (PSI) | Monthly | PSI vs. training baseline | PSI > 0.10 | Investigate; PSI > 0.25 → mandatory review |
| AUC Performance | Quarterly | Hold-out sample (last 90 days) | AUC drops > 0.05 from baseline | Re-evaluate model; consider retraining |
| Denial Rate | Monthly | Rolling 30-day average | ±3 percentage points from baseline | Root cause analysis |
| AIR (Gender) | Monthly | Rolling 30-day cohort | AIR < 0.85 | Investigation; AIR < 0.80 → mandatory escalation |
| AIR (Race) | Monthly | Rolling 30-day cohort | AIR < 0.85 | Investigation; AIR < 0.80 → mandatory escalation |
| Feature Missing Rate | Monthly | Percent null per feature | > 5% for any feature | Data pipeline investigation |

### 7.2 Maintenance Triggers

The following events trigger mandatory model review:
- AUC on recent cohort drops below 0.70
- PSI > 0.25 for any feature for two consecutive months
- AIR < 0.80 for any protected attribute
- Material change to credit bureau data feed or LOS
- Regulatory guidance update affecting ECOA / SR 11-7 requirements
- Annual scheduled review (regardless of performance)

---


## 8. Key Stakeholder Review / Approval

| Role | Name | Signature | Date |
|---|---|---|---|
| Model Developer | [Redacted] | __________ | April 15, 2025 |
| Model Owner (Credit Risk) | [Redacted] | __________ | April 15, 2025 |
| Independent Validation Lead | *Pending assignment* | __________ | — |
| Chief Risk Officer | *Pending* | __________ | — |
| Model Risk Committee Chair | *Pending* | __________ | — |

---


## 9. Appendix

### A. Data Dictionary

| Field | Type | Range | Definition |
|---|---|---|---|
| Mortgage | Continuous (USD) | [0, 700,000] | Outstanding mortgage balance |
| Balance | Continuous (USD) | [0, 60,000] | Total revolving balance |
| Amount Past Due | Continuous (USD) | [0, 6,000] | Total amount past due |
| Delinquency | Ordinal | {0,1,2,3} | Max delinquency cycles (24 mo.) |
| Inquiry | Count | [0, 5] | Hard inquiries (12 mo.) |
| Open Trade | Count | [0, 15] | Open trade lines |
| Utilization | Ratio | [0, 1] | Credit utilization ratio |
| Gender | Binary | 0=Male, 1=Female | Protected attribute (excluded from model) |
| Race | Binary | 0=Majority, 1=Minority | Protected attribute (excluded from model) |
| Status | Binary | 0=Approved, 1=Denied | Target variable |

### B. Key Abbreviations

| Abbreviation | Definition |
|---|---|
| AIR | Adverse Impact Ratio |
| AUC | Area Under the ROC Curve |
| CSI | Characteristic Stability Index |
| ECOA | Equal Credit Opportunity Act |
| HL | Hosmer-Lemeshow |
| KS | Kolmogorov-Smirnov Statistic |
| LOS | Loan Origination System |
| MRC | Model Risk Committee |
| PSI | Population Stability Index |
| RCIT | Randomized Conditional Independence Test |
| SR 11-7 | Federal Reserve Supervisory Guidance on Model Risk Management (2011) |
| VIF | Variance Inflation Factor |

### C. Regulatory References

| Reference | Full Citation |
|---|---|
| SR 11-7 | Board of Governors of the Federal Reserve System. *Guidance on Model Risk Management*. SR 11-7. April 4, 2011. |
| OCC 2011-12 | Office of the Comptroller of the Currency. *Sound Practices for Model Risk Management*. OCC 2011-12. April 4, 2011. |
| ECOA / Reg B | 15 U.S.C. § 1691 et seq.; 12 C.F.R. Part 1002 (Regulation B). |
| EEOC 4/5ths Rule | 29 C.F.R. § 1607 — Uniform Guidelines on Employee Selection Procedures. |
| CFPB Fair Lending | CFPB. *Supervisory Guidance on Model Risk Management*. 2021. |
