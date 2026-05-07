# Heart Disease Risk Prediction Project Report

## 1. Project Introduction

This project focuses on predicting heart disease risk using machine learning. The goal is to build a classification model that can predict whether a patient may have heart disease based on medical features such as age, chest pain type, blood pressure, cholesterol, maximum heart rate, and other clinical values.

This project was created as a machine learning portfolio project. It demonstrates data cleaning, exploratory data analysis, model training, model comparison, model saving, and deployment using a Streamlit web application.

---

## 2. Problem Statement

Heart disease is one of the major health concerns worldwide. Early risk prediction can help identify patients who may need further medical attention.

The main problem in this project is:

> Can we predict whether a patient has heart disease risk based on available clinical features?

This is a binary classification problem.

| Target Value | Meaning |
|---|---|
| 0 | No Heart Disease |
| 1 | Heart Disease Risk |

---

## 3. Dataset

The dataset used in this project is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

The raw dataset contained 303 records and 14 columns, including 13 input features and 1 target column.

After cleaning, the final dataset contained 297 records.

---

## 4. Data Cleaning

The dataset had missing values represented by `?`. These were converted into proper missing values using:

```python
df = pd.read_csv(
    data_path,
    names=column_names,
    na_values="?"
)
```

The main cleaning steps were:

- Added column names
- Converted `?` into missing values
- Removed missing values
- Removed duplicate rows
- Converted the target column into binary format
- Converted integer-like columns into integer data type
- Saved the cleaned dataset as `heart_cleaned.csv`

---

## 5. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset before model training.

Important EDA findings:

- The dataset was fairly balanced.
- 160 patients had no heart disease.
- 137 patients had heart disease.
- The average patient age was around 54.5 years.
- Chest pain type was strongly related to heart disease.
- Asymptomatic patients had a high number of heart disease cases.
- Important correlated features included `thal`, `ca`, `cp`, `oldpeak`, and `thalach`.

---

## 6. Models Trained

Three machine learning models were trained:

1. Logistic Regression
2. Decision Tree using Entropy
3. XGBoost Classifier

Logistic Regression was used as the baseline model.  
Decision Tree was used because it is interpretable and connects with entropy and information gain.  
XGBoost was used because it is a powerful ensemble learning model based on boosted decision trees.

---

## 7. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

For this project, recall is important because a false negative means the model missed a patient who actually had heart disease.

---

## 8. Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8333 | 0.8462 | 0.7857 | 0.8148 |
| Decision Tree | 0.7667 | 0.8500 | 0.6071 | 0.7083 |
| XGBoost | 0.8500 | 0.8800 | 0.7857 | 0.8302 |

---

## 9. Final Model Selection

XGBoost was selected as the final model because it achieved the best overall performance.

Final XGBoost results:

| Metric | Result |
|---|---:|
| Accuracy | 85.00% |
| Precision | 88.00% |
| Recall | 78.57% |
| F1-score | 83.02% |

XGBoost had the highest accuracy, precision, and F1-score among the tested models.

---

## 10. Confusion Matrix Explanation

The XGBoost confusion matrix was:

|  | Predicted No Heart Disease | Predicted Heart Disease |
|---|---:|---:|
| Actual No Heart Disease | 29 | 3 |
| Actual Heart Disease | 6 | 22 |

This means:

- 29 patients were correctly predicted as no heart disease.
- 22 patients were correctly predicted as heart disease.
- 3 patients were false positives.
- 6 patients were false negatives.

The false negatives are important because they represent patients who actually had heart disease but were predicted as low risk.

---

## 11. Feature Importance

The most important features in the final XGBoost model were:

- thal
- cp
- ca
- slope
- oldpeak

These features contributed most to the model’s predictions.

---

## 12. Streamlit App

A Streamlit web app was created to make the model interactive.

The app includes:

- Patient input panel
- Prediction button
- Risk probability
- Risk level
- Data sent to model
- Feature meaning guide
- Model performance summary
- Medical disclaimer
- Footer with contact links

The app loads the saved XGBoost model and feature names, accepts patient inputs, arranges features in the correct order, and returns the prediction.

---

## 13. Limitations

This project has some limitations:

- The dataset is small.
- The model should not be used for real diagnosis.
- The app is for educational and portfolio purposes only.
- More clinical validation would be needed for real medical use.
- The model was trained only on the Cleveland dataset.

---

## 14. Future Improvements

Future improvements can include:

- Hyperparameter tuning
- Cross-validation
- ROC-AUC curve
- Threshold tuning to improve recall
- SHAP explainability
- Online deployment
- More medical feature explanations
- A training script to regenerate model files
- Model versioning

---

## 15. Conclusion

This project successfully built a heart disease risk prediction system using machine learning. Logistic Regression, Decision Tree, and XGBoost models were trained and compared. XGBoost was selected as the final model because it achieved the best overall performance.

The project also includes a Streamlit app that allows users to input patient information and receive a risk prediction. Overall, this project demonstrates a complete machine learning workflow from data cleaning to model deployment.