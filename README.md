# ML-health-analysis
# Predicting Health Outcomes with Machine Learning

This project explores how machine learning models can predict self-reported health status using the Behavioral Risk Factor Surveillance System (BRFSS) data from the CDC. It showcases a full data science pipeline including data cleaning, feature engineering, model training, and interpretation.

## 📊 Project Overview

- **Dataset**: BRFSS 2015 survey data (sampled to 100,000 rows for efficiency). The dataset is not uploaded due to size limit, but can be found at:https://www.cdc.gov/brfss/annual_data/annual_data.htm
- **Target Variable**: **`_RFHLTH`** – a self-reported rating of general health (1 = Good/Better Health and 2 = Fair/Poor Health)
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

## 📈 Results and Discussions

## Logistic Regression
pseudo R-squared: 0.364



  
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

***Table 1: Top 5 Positive/Negative Variables for Logistic Regressions***

### Interpretation of Logistic Regression Results

The results make intuitive sense — most of the important variables relate to participants' physical condition or disease history. For example:

- **`QLACTLM2`**:  
  Asks whether the participant is limited in any way due to physical or mental conditions.  
  → Answering `'yes'` decreases the odds of reporting good health by:  
  **`1 - e^(-0.768) ≈ 53.6%`**.

- **`EXERANY2`**:  
  Asks whether the participant engaged in any physical activity (e.g., jogging) in the past month.  
  → Answering `'yes'` increases the odds of reporting good health by:  
  `e^(0.366) - 1 ≈ 44%`.

### Additional Observations

- **`EDUCA`** (education level) and **`INTERNET`** (internet usage in the past month) are also positively associated with self-reported good health.
  - One might speculate that higher education and regular internet use improve access to health-related information, encouraging healthier behaviors.
  - However, this is only a hypothesis, the logistic regression model alone does not provide causal inference.

### Limitation of Logistic Regression

A major drawback of logistic regression in this context is its inability to handle missing values (`NaN`).  
As a result:
- Any conditional variable (i.e., only asked of a subset of respondents) must be omitted to run the model.
- This could potentially exclude informative predictors.

## Decision Tree

![AUC Curve - Decision Tree](images/dt1.png)

***Figure 1: Visualisation of Decision Tree Branches***

![AUC Curve - Decision Tree](images/dt2.png)

***Figure 2: Confusion Matrix for Decison Tree Model***


### Interpretation of Decision Tree Results

The confusion matrix shows that around `87%` of results are correctly predicted using this model. It is important to note that we have an imbalanced dataset where the majority of individuals report good/better health. In this case, accuracy is not the best measure for model performance, we have to introduce the ROC-AUC curve as a more robust indicator.

## Random Forest

![AUC Curve - Random Forest](images/rf1.png)

***Figure 3: Confusion Matrix for the Initial Random Forest Model***

### Random Forest vs. Decision Tree

We compute the confusion matrix of a Random Forest model (n_estimators = 100, max_depth = None, max_features = 'sqrt'), we can see that it slightly outperforms the Decision Tree model.
This result is expected since Random Forest uses multiple trees to reduce overfitting and stablize prediction.

### Model Tuning

![AUC Curve - Decision Tree](images/rf2.png)

***Figure 4: Performance Comparison of Different Random Forest Models***

We then test different combinations of model parameters to find the best performing model. From the figure, we can see that fixing other variables, model performance tends to increase with number of estimators. We also find that:

**`(n_estimators = 300, max_depth = 20, max_features = 'sqrt')`**

gives the highest ROC AUC at **`0.893`**.



### Feature Importance

We then proceed to generate the list of the 10 most important features.

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

***Table 2: 10 Most Important Features in Best Random Forest Model***

## XGBoost

### Model Tuning

![AUC Curve - Random Forest](images/xg1.png)

***Figure 5: Performance Comparison of Different XGBoost Models***

We iterate the same process for XGBoost models and find that

**`(n_estimators = 300, max_depth = 5, learning_rate = 0.1)`** gives the highest ROC AUC at 0.900.

Another interesting observation is that, while most of the time model performance increases with number of estimators (fixing the rest of course), in the special case where learning rate = `0.1`, ROC AUC actually moves in an opposite direction. This is because a higher learning rate gives each iteration a larger update, which also introduces potential overfitting.
### Feature Importance

We then proceed to generate the list of the 10 most important features.
 
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

***Table 3: 10 Most Important Features in Best XGBoost Model***

## 🚀Final Summary

### Model Comparison

| Model           | Mean AUC | Std. AUC |
|-----------------|----------|----------|
| XGBoost         | 0.901    | 0.003    |
| Random Forest   | 0.893    | 0.003    |
| Decision Tree   | 0.846    | 0.003    |

***Table 4: Top Models Performance Comparison***

Table 4 shows that XGBoost > Random Forest > Decision Tree in terms of performance.

This is a rather intuitive result since both XGBoost and Random Forest uses multiple decision trees for modelling.

Taking one stepn further, XGBoost is expected to outperform Random Forest because of its direct on error correction (will be further discussed in data insight).

### Data Insights

Comparing most impactful features between the three models using Table 1-3, **we can see that**:

- **`QLACTLM2`** and **`DIFFWALK`**:
  appears in all three tables
  

- **`INTERNET`**, **`EXERANY2`**, **`EDUCA`**, **`ARTHSOCL`**, and **`PHYSHLTH`**:
  appears in 2 out of 3 tables
  

The contrast in feature importance in the three models is mainly due to **difference in approach**:
- Logistic Regression assigns importance using MLE (Maximum Likelihood Estimator)
- Random Forest creates independent trees and takes into account the 'opinion' of different trees.
- XGBoost builds trees sequentially, with the newer one trying to correct errors of previous trees.

Overall, the results appear to convincing.

- **Physical Limitations**:
  - **`QLACTLM2`** has been discussed in the Logistic Regression section.
  - **`DIFFWALK`** asks if participants have difficulty walking, this appears to be a determining factor of their self-reported health condition.
  - **`PHYSHLTH`** asks how many days have participants been physically unwell in the past month, make
- **`ARTHSOCL`**:
  - asks how many days has arthritis interfered with normal social activities (e.g. going shopping)
  - **Why is arthritis considered more important when questions related to high blood pressure, heart disease, different types of cancer is asked?**
    
    There is a possibility that people weigh arthritis more because of its direct effect of movement impairment. However, I strongly suspect that the biasness of the question lead to such result. In the codebook, the original question tied to **`ARTHSOCL`** is **`During the past 30 days, to what extent has your arthritis or joint symptoms interfered with your normal social activities, such as going shopping, to the movies, or to religious or social gatherings?`**. This is the only servey question of that year that ties an illness to social activities. Therefore, **`ARTHSOCL`** having a higher feature importance actually reflects how interference of normal social activities affects self-reported health.
    
- **`INTERNET`** and **`EDUCA`**:
  - appears not only in the Logistic Regression 
  - while none of the model provides causal inference, having the two variables appear in more than one model speaks credibility
  - would be interesting understanding why they are important. Is the theory of broader health information access valid?
  
  





