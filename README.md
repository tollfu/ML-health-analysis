# ML-health-analysis
# Predicting Health Outcomes with Machine Learning

This project explores how machine learning models can predict self-reported health status using the Behavioral Risk Factor Surveillance System (BRFSS) data from the CDC. It showcases a full data science pipeline including data cleaning, feature engineering, model training, and interpretation.

## 📊 Project Overview

- **Dataset**: BRFSS 2015 survey data (sampled to 100,000 rows for efficiency). The dataset is not uploaded due to size limit, but can be found at:https://www.cdc.gov/brfss/annual_data/annual_data.htm
- **Target Variable**: `_RFHLTH` – a self-reported rating of general health (1 = Good/Better Health and 2 = Fair/Poor Health)
- **Models Used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest Classifier
  - XGBoost

The objective is to predict a respondent’s self reported general health based on other numeric health and behavior-related variables. Then compare performance of model precision to find the best model.

## 🔍 Key Steps

- **Data Cleaning and Manipulation**:
  - removal of irrelevant and less important features
  - feature recoding (e.g. conversion of 1/2 -> 0/1, filling in NaN values by averaging)
  - identification of universal/conditional features and separate engineering
  
- **Testing and Fine Tuning of Multiple Models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest (varying number of estimators, maximum depth and maximum features)
  - XGBoost (varying number of estimators, maximum depth and learning rate)

- **Performance Comparison and Insight Extraction**:
  - graph the AUC-ROC curve
  - list and cross-match feature importance

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


| Rank | Feature         | Importance |
|------|------------------|------------|
| 1    | PHYSHLTH         | 0.0937     |
| 2    | DIFFWALK         | 0.0453     |
| 3    | QLACTLM2         | 0.0361     |
| 4    | DIFFALON         | 0.0291     |
| 5    | ARTHSOCL         | 0.0216     |
| 6    | _BMI5            | 0.0193     |
| 7    | MENTHLTH         | 0.0188     |
| 8    | USEEQUIP         | 0.0187     |
| 9    | JOINPAIN         | 0.0185     |
| 10   | _VEGESUM         | 0.0168     |

| Rank | Feature         | Importance |
|------|------------------|------------|
| 1    | PHYSHLTH         | 0.1271     |
| 2    | QLACTLM2         | 0.1145     |
| 3    | INTERNET         | 0.0740     |
| 4    | DIFFWALK         | 0.0557     |
| 5    | ARTHSOCL         | 0.0456     |
| 6    | _INCOMG          | 0.0421     |
| 7    | EXERANY2         | 0.0413     |
| 8    | BPHIGH4          | 0.0338     |
| 9    | EDUCA            | 0.0329     |
| 10   | DECIDE           | 0.0284     |


| Model           | Mean AUC | Std. AUC |
|-----------------|----------|----------|
| XGBoost         | 0.901    | 0.003    |
| Random Forest   | 0.893    | 0.003    |
| Decision Tree   | 0.846    | 0.003    |
