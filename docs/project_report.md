# Explainable AI Diabetes Prediction System — Project Report

## 1. Introduction
This project demonstrates diabetes risk prediction using supervised machine learning and Explainable AI.

## 2. Objectives
- Preprocess a diabetes dataset.
- Compare KNN, SVM, and Random Forest.
- Evaluate models using multiple classification metrics.
- Provide an interactive prediction interface.
- Explain individual predictions using SHAP.

## 3. Dataset
Dataset file: `data/diabetes.csv`

Input features:
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.

Target:
Outcome.

## 4. Methodology
Data preprocessing → train/test split → model training → evaluation → best-model selection → patient prediction → SHAP explanation.

## 5. Models
KNN, SVM, and Random Forest.

## 6. Evaluation
Accuracy, Precision, Recall, F1 Score, and ROC-AUC are calculated on the test set.

## 7. Explainable AI
SHAP is used to identify the contribution of individual features to a prediction.

## 8. Interface
The application is implemented using Streamlit and Plotly.

## 9. Limitations
This is an educational machine-learning project and is not a clinical diagnostic system. Results depend on the dataset and model assumptions.

## 10. Future Work
Hyperparameter optimization, cross-validation, model persistence, testing, deployment, and expanded explainability.
