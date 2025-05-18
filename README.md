# ML-health-analysis
# Predicting Health Outcomes with Machine Learning

This project explores how machine learning models can predict self-reported health status using the Behavioral Risk Factor Surveillance System (BRFSS) data from the CDC. It showcases a full data science pipeline including data cleaning, feature engineering, model training, and interpretation.

## 📊 Project Overview

- **Dataset**: BRFSS 2015 survey data (sampled to 100,000 rows for efficiency)
- **Target Variable**: `GENHLTH` – a self-reported rating of general health (1 = Excellent to 5 = Poor)
- **Models Used**:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost

The objective is to predict a respondent’s general health based on other numeric health and behavior-related variables.

## 🔍 Key Steps

- Filtered numeric variables and dropped high-missing-value columns
- Removed constant and missing-value rows
- Performed train/test split
- Trained models using `scikit-learn`
- Evaluated performance with accuracy and confusion matrices
- Visualized top predictors using feature importance (Random Forest)

## 📈 Results Summary

- **Logistic Regression Accuracy**: ~X.XX (fill in actual result)
- **Random Forest Accuracy**: ~X.XX (fill in actual result)
- **XGBoost Accuracy**: ~X.XX (fill in actual result)
- **Top Predictive Features**: (e.g., BMI, number of mentally unhealthy days, etc.)

Random Forest outperformed Logistic Regression and identified the most important features contributing to health status classification.
XGBoost outperformed Logistic Regression and identified the most important features contributing to health status classification.
## 📁 Repository Structure
| Feature   |   Coefficient |   Std Error |   P-value | Direction   |
|:----------|--------------:|------------:|----------:|:------------|
| _RFHYPE5  |         0.388 |       0.173 |     0.025 | Positive    |
| EXERANY2  |         0.366 |       0.165 |     0.026 | Positive    |
| DIABETE3  |         0.291 |       0.013 |     0     | Positive    |
| EDUCA     |         0.285 |       0.023 |     0     | Positive    |
| INTERNET  |         0.261 |       0.028 |     0     | Positive    |
| QLACTLM2  |        -0.768 |       0.027 |     0     | Negative    |
| CHCCOPD1  |        -0.548 |       0.034 |     0     | Negative    |
| CHCOCNCR  |        -0.542 |       0.031 |     0     | Negative    |
| CHCKIDNY  |        -0.508 |       0.047 |     0     | Negative    |
| DIFFWALK  |        -0.466 |       0.03  |     0     | Negative    |



