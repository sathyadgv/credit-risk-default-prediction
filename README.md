
# Credit Risk Default Prediction

## Overview

This project develops an end-to-end credit risk classification workflow to predict the probability of loan default.

The objective was to compare an interpretable **Logistic Regression** model with a more flexible **XGBoost** classifier and evaluate their ability to discriminate between defaulting and non-defaulting borrowers.

The project covers the complete modelling process from exploratory data analysis and data cleaning to model evaluation using both standard machine-learning metrics and credit-risk-specific measures such as **Gini** and **Kolmogorov-Smirnov (KS)**.

## Dataset

The project uses the **Credit Risk Dataset** available on Kaggle.

The dataset contains approximately 32,000 loan observations and includes borrower and loan characteristics such as:

* Age
* Income
* Employment length
* Home ownership
* Loan purpose
* Loan amount
* Interest rate
* Loan grade
* Loan-to-income ratio
* Previous default history
* Credit history length

The target variable is `loan_status`:

* `0` = No default
* `1` = Default

The observed default rate in the dataset is approximately **21.8%**.

## Project Workflow

### 1. Data Exploration and Quality Analysis

The dataset was inspected for:

* Missing values
* Implausible observations
* Outliers
* Class imbalance
* Numerical and categorical variable distributions
* Default rates across borrower and loan characteristics

Several implausible observations were identified, including borrower ages above 100 years and employment lengths incompatible with borrower age.

### 2. Exploratory Credit Risk Analysis

Default behaviour was analysed across several characteristics.

Some of the strongest patterns included:

* Higher default rates among borrowers renting their homes
* Increasing default rates across worse loan grades
* Higher default rates for borrowers with previous defaults
* Higher loan-to-income ratios among defaulting borrowers
* Higher interest rates among defaulting borrowers

### 3. Data Preparation

The modelling workflow included:

* Train/test split with stratification
* Median imputation for missing numerical values
* One-hot encoding of categorical variables
* Feature scaling for Logistic Regression
* Prevention of data leakage by learning preprocessing parameters exclusively from the training data

The final modelling dataset contained **22 predictive features** after encoding.

## Logistic Regression

Logistic Regression was used as an interpretable baseline model.

### Performance

| Metric            | Result |
| ----------------- | -----: |
| ROC-AUC           |  0.844 |
| Gini              |  0.689 |
| KS                |  0.569 |
| Default Precision |   0.73 |
| Default Recall    |   0.45 |
| Default F1-score  |   0.56 |
| Accuracy          |   0.84 |

Threshold analysis showed the expected trade-off between precision and recall.

For example, reducing the classification threshold from `0.50` to approximately `0.30` increased default recall substantially, although at the cost of lower precision.

The maximum KS statistic occurred around a threshold of **0.272**.

## XGBoost

An XGBoost classifier was subsequently trained using the same train/test sample.

### Performance

| Metric            |    Result |
| ----------------- | --------: |
| ROC-AUC           | **0.925** |
| Gini              | **0.850** |
| KS                | **0.696** |
| Default Precision |  **0.91** |
| Default Recall    |  **0.70** |
| Default F1-score  |  **0.79** |
| Accuracy          |  **0.92** |

XGBoost significantly outperformed Logistic Regression across all major predictive performance metrics.

## Model Comparison

| Metric            | Logistic Regression |   XGBoost |
| ----------------- | ------------------: | --------: |
| ROC-AUC           |               0.844 | **0.925** |
| Gini              |               0.689 | **0.850** |
| KS                |               0.569 | **0.696** |
| Default Precision |                0.73 |  **0.91** |
| Default Recall    |                0.45 |  **0.70** |
| Default F1-score  |                0.56 |  **0.79** |
| Accuracy          |                0.84 |  **0.92** |

**XGBoost was selected as the best-performing model**, while Logistic Regression remains valuable as a more transparent and interpretable benchmark.

## Feature Importance

The most important XGBoost predictors included:

1. `loan_percent_income`
2. `person_home_ownership_RENT`
3. `loan_int_rate`
4. `loan_grade_D`
5. `loan_grade_C`
6. `person_income`

The results suggest that borrower repayment capacity, loan pricing and existing credit-risk indicators provide substantial discriminatory information.

## Credit Risk Metrics

In addition to standard classification metrics, the models were evaluated using two measures commonly associated with credit risk model discrimination:

### Gini

The Gini coefficient was derived from ROC-AUC:

`Gini = 2 × AUC - 1`

XGBoost achieved a Gini coefficient of approximately **0.850**, compared with **0.689** for Logistic Regression.

### Kolmogorov-Smirnov Statistic

The KS statistic measures the maximum separation between the cumulative behaviour of defaults and non-defaults.

XGBoost achieved a maximum KS of approximately **0.696**, compared with **0.569** for Logistic Regression.

## Limitations

This project should be interpreted as an educational credit risk modelling exercise rather than a production-ready credit decision system.

Important limitations include:

* No external or out-of-time validation was performed.
* Cross-validation was not included in the final version.
* `loan_grade` may contain information derived from a previous underwriting process and should therefore be reviewed for potential data leakage depending on the intended prediction point.
* The dataset does not contain sufficient information to build complete PD-LGD-EAD or Expected Loss models.
* Model fairness, stability and population drift were outside the scope of this first project.

## Technologies

* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* XGBoost
* Jupyter Notebook

## Conclusion

This project demonstrates an end-to-end credit risk modelling workflow, combining data quality analysis, exploratory analysis, preprocessing, statistical classification and machine-learning modelling.

The results show that XGBoost provides substantially stronger predictive discrimination than the Logistic Regression baseline, achieving a **ROC-AUC of 0.925, Gini of 0.850 and KS of 0.696**.

At the same time, the project illustrates the importance of balancing predictive performance with interpretability, threshold selection and data quality considerations in credit risk modelling.
