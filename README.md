# Predicting-Diabetic-Patient-Readmission-Using-Logistic-Regression

This project focuses on developing predictive models for identifying diabetic patients at risk of hospital readmission using real-world healthcare data. It evaluates baseline logistic regression against regularized variants (L2 and ElasticNet), and implements a complete machine learning pipeline including feature engineering, multicollinearity checks, outlier handling, and hyperparameter optimization.

## 📊 Problem Statement
Hospital readmissions among diabetic patients represent a significant challenge to healthcare systems by increasing costs and indicating treatment gaps. This project aims to classify patients based on their likelihood of being readmitted within 30 days, leveraging machine learning for early intervention and resource optimization.

## 🧾 Dataset
The dataset used (`hospitaldata.csv`) contains **101,763 patient encounters** across various hospital settings with 47 features including:
- Demographics (age, gender, race)
- Medical details (diagnosis codes, lab results, medication history)
- Encounter information (admission type, discharge disposition, readmission status)

The target variable is transformed into a binary classification: **Readmitted vs. Not Readmitted**.

## 🔍 Key Steps & Methodology

### 1. Data Preprocessing
- Handling missing values and noisy categories (`Unknown`, `Other`)
- Transformation of ICD-9 codes into clinical groups
- Ordinal encoding for age, A1C results, and glucose levels
- Outlier handling using IQR for numeric features

### 2. Feature Selection
- **Chi-squared test** for categorical variable relevance
- **Variance Inflation Factor (VIF)** for multicollinearity among numeric features

### 3. Modeling
- **Model 1:** Baseline Logistic Regression (with class weighting)
- **Model 2:** L2-Regularized Logistic Regression (Ridge)
- **Model 3:** ElasticNet Logistic Regression (L1 + L2)
- **Model 4:** ElasticNet with GridSearchCV (hyperparameter tuning)

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Curve comparison

## 🧪 Results Summary
- All models achieved comparable performance
- **Model 2 (L2-Regularization)** performed slightly better in terms of **recall** and **F1-score**
- GridSearch hyperparameter tuning in Model 4 yielded minimal improvement due to model stability with regularization

## 📈 Performance Metrics
| Model                       | Accuracy | Recall | F1-Score | ROC-AUC  |
|-----------------------------|----------|--------|----------|----------|
| Logistic Regression         | 62.7%    | 0.564  | 0.584    | 0.670    |
| L2-Regularized (Ridge)      | ~63%     | 0.565  | 0.585    | 0.671    |
| ElasticNet                  | ~63%     | ~0.565 | ~0.585   | 0.671    |
| ElasticNet + GridSearchCV   | ~62.5%   | ~0.563 | 0.583    | 0.669    |

## ⚙️ Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
## 📘 Jupyter Notebook

The complete model code and data processing pipeline can be found in  
[`Readmission_Prediction_Model.ipynb`](./Code_File.ipynb)

## 📚 References
- [ICD-9 Codes - Wikipedia](https://en.wikipedia.org/wiki/List_of_ICD-9_codes)
- [Logistic Regression - GeeksforGeeks](https://www.geeksforgeeks.org/understanding-logistic-regression/)
- [EDA Guide - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/)
- [IEEE Xplore Research on Diabetes Prediction](https://ieeexplore.ieee.org/document/8342938)
