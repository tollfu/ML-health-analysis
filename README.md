# ML-health-analysis
# Predicting Health Outcomes with Machine Learning

This project explores how machine learning models can predict self-reported health status using the Behavioral Risk Factor Surveillance System (BRFSS) data from the CDC. It showcases a full data science pipeline including data cleaning, feature engineering, model training, and interpretation.

## 📊 Project Overview

- **Dataset**: BRFSS 2015 survey data (sampled to 100,000 rows for efficiency)
- **Target Variable**: `GENHLTH` – a self-reported rating of general health (1 = Excellent to 5 = Poor)
- **Models Used**:
  - Logistic Regression
  - Random Forest Classifier

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
- **Top Predictive Features**: (e.g., BMI, number of mentally unhealthy days, etc.)

Random Forest outperformed logistic regression and identified the most important features contributing to health status classification.

## 📁 Repository Structure

health-ml-project/
├── data/
│ └── sample_brfss.csv # Sample dataset (1000 rows)
├── Health_ML_Results.ipynb # Full notebook with EDA + modeling
├── requirements.txt # Dependencies
├── README.md # Project overview

