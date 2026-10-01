# 🧠 Explainable AI Diabetes Prediction System

An interactive machine learning application for diabetes risk prediction using multiple supervised learning algorithms and Explainable AI (SHAP).

> **Educational project only. This application is not a medical diagnostic tool.**

## 🚀 Project Overview

This project demonstrates an end-to-end machine learning workflow:

- Data preprocessing and missing-value handling
- Feature scaling where required
- Training multiple classification models
- Model performance comparison
- Automatic selection of the model with the highest ROC-AUC
- Interactive diabetes-risk prediction
- Probability and risk-level visualization
- Confusion matrix and ROC curve
- Feature importance
- SHAP-based individual prediction explanations
- Interactive Streamlit dashboard

## 🤖 Machine Learning Models

The application compares:

1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVM)**
3. **Random Forest**

The dataset is divided into training and testing sets using an 80/20 split with stratification.

## 🧹 Data Preprocessing

The following values are treated as missing in the dataset:

- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI

Missing values are handled using median imputation.

KNN and SVM use StandardScaler inside a scikit-learn Pipeline.

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The application displays the results in both a table and interactive Plotly visualization.

## 🔍 Explainable AI

The application uses **SHAP (SHapley Additive exPlanations)** to show how individual input features contribute to a prediction.

The dashboard displays:

- SHAP values
- Feature contribution chart
- Most influential features
- Direction of contribution toward the predicted probability

This helps demonstrate the concept of interpretable machine learning rather than only showing a prediction.

## 🩺 Patient Prediction

Users can enter:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The system then displays:

- Predicted class
- Diabetes probability
- Low / Moderate / High risk level
- Probability gauge
- SHAP explanation

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- SHAP
- Plotly

## 📁 Project Structure

```text
Explainable-AI-Diabetes-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── diabetes.csv
│
├── screenshots/
│   ├── dashboard.png
│   ├── model_comparison.png
│   ├── prediction.png
│   └── shap_explanation.png
│
└── docs/
    └── project_report.pdf
```

The `screenshots/` and `docs/` folders are optional placeholders. Add your actual screenshots/report before publishing if you have them.

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Explainable-AI-Diabetes-Prediction-System.git
cd Explainable-AI-Diabetes-Prediction-System
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📈 Application Workflow

```text
Diabetes Dataset
       ↓
Data Preprocessing
       ↓
Train/Test Split
       ↓
 ┌─────┼─────────────┐
 ↓     ↓             ↓
KNN   SVM      Random Forest
 └─────┼─────────────┘
       ↓
Model Evaluation
       ↓
ROC-AUC Comparison
       ↓
Best Model Selection
       ↓
Patient Input
       ↓
Prediction + Probability
       ↓
SHAP Explanation
```

## ⚠️ Disclaimer

This project is intended for educational and demonstration purposes. Model predictions should not be interpreted as medical diagnoses or used as a substitute for professional medical advice.

## 👨‍💻 Author

**Muhammad Afaq**

BSAI Student

GitHub: `https://github.com/YOUR-USERNAME`

## ⭐ Future Improvements

Possible future extensions include:

- Hyperparameter optimization
- Cross-validation-based model comparison
- Additional diabetes datasets
- Model persistence instead of retraining on application startup
- User authentication
- Prediction history
- Cloud deployment
- Automated testing
- More advanced SHAP visualizations
- Experiment tracking
