# Heart Disease Risk Prediction Using Decision Trees and XGBoost

A machine learning portfolio project that predicts heart disease risk using patient medical data.  
This project covers key ML concepts such as data cleaning, exploratory data analysis, classification, Logistic Regression, Decision Trees, entropy, XGBoost, model evaluation, and Streamlit app development.

---

## Project Overview

The goal of this project is to predict whether a patient may have heart disease based on medical features such as age, chest pain type, resting blood pressure, cholesterol, maximum heart rate, exercise-induced angina, and other clinical attributes.

This is a binary classification project:

| Target Value | Meaning |
|---|---|
| 0 | No Heart Disease |
| 1 | Heart Disease Risk |

The original dataset had target values from 0 to 4. For this project, values 1, 2, 3, and 4 were converted into 1, meaning heart disease is present.

---

## Dataset Information

The dataset used in this project is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

### Dataset Summary

| Item | Description |
|---|---|
| Dataset | Cleveland Heart Disease Dataset |
| Source | UCI Machine Learning Repository |
| Original Records | 303 |
| Cleaned Records | 297 |
| Input Features | 13 |
| Target Column | target |
| Problem Type | Binary Classification |

### Features Used

| Feature | Meaning |
|---|---|
| age | Patient age |
| sex | Patient sex |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar above 120 mg/dl |
| restecg | Resting ECG result |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression value |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia test category |
| target | Heart disease result |

---

## Data Cleaning

The raw dataset contained missing values represented by `?`. These values were converted into proper missing values using:

```python
na_values="?"