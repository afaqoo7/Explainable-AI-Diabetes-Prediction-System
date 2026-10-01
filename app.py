import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import shap

import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable AI Disease Prediction",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 25px;
    }

    .risk-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"

DATA_PATH = "data/diabetes.csv"


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Explainable AI Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Diabetes Prediction using Machine Learning and Explainable AI</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    data = pd.read_csv(DATA_PATH)

    return data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(data):

    data = data.copy()

    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in zero_as_missing:

        data[column] = data[column].replace(
            0,
            np.nan
        )

    X = data[FEATURES]

    y = data[TARGET]

    return X, y


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    models = {

        "KNN": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance"
                )
            )
        ]),

        "SVM": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                SVC(
                    probability=True,
                    kernel="rbf",
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ])
    }

    return models


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models():

    data = load_dataset()

    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = create_models()

    results = []

    trained_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        })

        trained_models[name] = model

    results_df = pd.DataFrame(
        results
    )

    best_model_name = (
        results_df
        .sort_values(
            "ROC-AUC",
            ascending=False
        )
        .iloc[0]["Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    return (
        data,
        X_train,
        X_test,
        y_train,
        y_test,
        trained_models,
        results_df,
        best_model_name,
        best_model
    )


# ============================================================
# START TRAINING
# ============================================================

try:

    (
        data,
        X_train,
        X_test,
        y_train,
        y_test,
        trained_models,
        results_df,
        best_model_name,
        best_model
    ) = train_models()

except FileNotFoundError:

    st.error(
        "diabetes.csv was not found."
    )

    st.info(
        "Make sure your dataset is located at: "
        "data/diabetes.csv"
    )

    st.stop()

except Exception as e:

    st.error(
        f"Error while training models: {e}"
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧠 Project Information"
)

st.sidebar.write(
    "This system compares multiple supervised "
    "machine learning algorithms and explains "
    "individual predictions using SHAP."
)

st.sidebar.divider()

st.sidebar.metric(
    "Dataset Records",
    len(data)
)

st.sidebar.metric(
    "Features",
    len(FEATURES)
)

st.sidebar.success(
    f"Best Model: {best_model_name}"
)

st.sidebar.divider()

st.sidebar.info(
    "Educational project only. "
    "This system is not a medical diagnostic tool."
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Records",
        len(data)
    )

with col2:

    st.metric(
        "Input Features",
        len(FEATURES)
    )

with col3:

    diabetic_count = int(
        data[TARGET].sum()
    )

    st.metric(
        "Diabetic",
        diabetic_count
    )

with col4:

    non_diabetic_count = int(
        (data[TARGET] == 0).sum()
    )

    st.metric(
        "Non-Diabetic",
        non_diabetic_count
    )


with st.expander(
    "📋 View Complete Dataset"
):

    st.dataframe(
        data,
        use_container_width=True
    )


st.divider()


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header(
    "🤖 Machine Learning Model Comparison"
)

display_results = results_df.copy()

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]:

    display_results[column] = (
        display_results[column] * 100
    ).round(2).astype(str) + "%"


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL PERFORMANCE CHART
# ============================================================

st.subheader(
    "Model Performance"
)

chart_data = results_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    chart_data,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Comparison of Machine Learning Models"
)

fig.update_yaxes(
    range=[0, 1]
)

fig.update_layout(
    legend_title="Metric"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.success(
    f"Selected Best Model: {best_model_name}"
)

st.caption(
    "The model with the highest ROC-AUC score is selected."
)


st.divider()


# ============================================================
# BEST MODEL EVALUATION
# ============================================================

st.header(
    "📈 Best Model Evaluation"
)

best_predictions = best_model.predict(
    X_test
)

best_probabilities = (
    best_model.predict_proba(
        X_test
    )[:, 1]
)


accuracy = accuracy_score(
    y_test,
    best_predictions
)

precision = precision_score(
    y_test,
    best_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    best_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    best_predictions,
    zero_division=0
)

auc_score = roc_auc_score(
    y_test,
    best_probabilities
)


col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

with col2:

    st.metric(
        "Precision",
        f"{precision:.2%}"
    )

with col3:

    st.metric(
        "Recall",
        f"{recall:.2%}"
    )

with col4:

    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )

with col5:

    st.metric(
        "ROC-AUC",
        f"{auc_score:.2%}"
    )


# ============================================================
# CONFUSION MATRIX + ROC CURVE
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# CONFUSION MATRIX
# ============================================================

with col1:

    st.subheader(
        "Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        best_predictions
    )

    confusion_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[
                "Non-Diabetic",
                "Diabetic"
            ],
            y=[
                "Non-Diabetic",
                "Diabetic"
            ],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues"
        )
    )

    confusion_fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        title="Confusion Matrix"
    )

    st.plotly_chart(
        confusion_fig,
        use_container_width=True
    )


# ============================================================
# ROC CURVE
# ============================================================

with col2:

    st.subheader(
        "ROC Curve"
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        best_probabilities
    )

    roc_fig = go.Figure()

    roc_fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {auc_score:.3f}"
        )
    )

    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(
                dash="dash"
            )
        )
    )

    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader(
    "Classification Report"
)

report = classification_report(
    y_test,
    best_predictions,
    target_names=[
        "Non-Diabetic",
        "Diabetic"
    ],
    output_dict=True
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


st.divider()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header(
    "🔍 Feature Importance"
)

if best_model_name == "Random Forest":

    rf_pipeline = (
        trained_models["Random Forest"]
    )

    rf_model = (
        rf_pipeline
        .named_steps["model"]
    )

    importance = (
        rf_model.feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature": FEATURES,

        "Importance": importance

    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        importance_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    sorted_importance = (
        importance_df
        .sort_values(
            "Importance"
        )
    )

    importance_fig = px.bar(
        sorted_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )

    st.plotly_chart(
        importance_fig,
        use_container_width=True
    )

else:

    st.info(
        "The best model is not Random Forest. "
        "Individual prediction explanations are still "
        "available below using SHAP."
    )


st.divider()


# ============================================================
# PATIENT INPUT
# ============================================================

st.header(
    "🩺 Patient Prediction"
)

st.write(
    "Enter patient information to generate a model prediction."
)


col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=2
    )

    glucose = st.number_input(
        "Glucose",
        min_value=1.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=1.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=1.0,
        max_value=100.0,
        value=25.0
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=1.0,
        max_value=900.0,
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        max_value=70.0,
        value=28.5
    )

    diabetes_function = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.01,
        max_value=3.0,
        value=0.35
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=35
    )


# ============================================================
# PATIENT DATAFRAME
# ============================================================

patient_data = pd.DataFrame({

    "Pregnancies": [
        pregnancies
    ],

    "Glucose": [
        glucose
    ],

    "BloodPressure": [
        blood_pressure
    ],

    "SkinThickness": [
        skin_thickness
    ],

    "Insulin": [
        insulin
    ],

    "BMI": [
        bmi
    ],

    "DiabetesPedigreeFunction": [
        diabetes_function
    ],

    "Age": [
        age
    ]
})


st.subheader(
    "Entered Patient Data"
)

st.dataframe(
    patient_data,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Diabetes Risk",
    type="primary",
    use_container_width=True
)


if predict_button:

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = best_model.predict(
        patient_data
    )[0]

    probability = (
        best_model
        .predict_proba(
            patient_data
        )[0][1]
    )

    probability_percentage = (
        probability * 100
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.30:

        risk_level = "Low"

    elif probability < 0.60:

        risk_level = "Moderate"

    else:

        risk_level = "High"


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header(
        "Prediction Result"
    )


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Higher Diabetes Risk"
            )

        else:

            st.success(
                "✅ Lower Diabetes Risk"
            )


    with result_col2:

        st.metric(
            "Diabetes Probability",
            f"{probability_percentage:.2f}%"
        )


    with result_col3:

        if risk_level == "High":

            st.error(
                f"Risk Level: {risk_level}"
            )

        elif risk_level == "Moderate":

            st.warning(
                f"Risk Level: {risk_level}"
            )

        else:

            st.success(
                f"Risk Level: {risk_level}"
            )


    # ========================================================
    # PROBABILITY VISUALIZATION
    # ========================================================

    st.subheader(
        "Diabetes Probability"
    )

    probability_fig = go.Figure(

        go.Indicator(
            mode="gauge+number",
            value=probability_percentage,
            title={
                "text": "Predicted Probability"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "#d4edda"
                    },
                    {
                        "range": [30, 60],
                        "color": "#fff3cd"
                    },
                    {
                        "range": [60, 100],
                        "color": "#f8d7da"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        probability_fig,
        use_container_width=True
    )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.divider()

    st.header(
        "🧠 Explainable AI"
    )

    st.write(
        """
        SHAP helps explain how individual features
        influenced this specific prediction.
        """
    )


    # ========================================================
    # SHAP
    # ========================================================

    try:

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        if best_model_name == "Random Forest":

            rf_pipeline = (
                trained_models["Random Forest"]
            )

            imputer = (
                rf_pipeline
                .named_steps["imputer"]
            )

            rf_model = (
                rf_pipeline
                .named_steps["model"]
            )

            transformed_patient = (
                imputer.transform(
                    patient_data
                )
            )

            explainer = (
                shap.TreeExplainer(
                    rf_model
                )
            )

            shap_result = explainer(
                transformed_patient
            )

            shap_values = shap_result.values

            if len(shap_values.shape) == 3:

                shap_values = (
                    shap_values[0, :, 1]
                )

            elif len(shap_values.shape) == 2:

                shap_values = (
                    shap_values[0]
                )

            else:

                shap_values = np.array(
                    shap_values
                ).flatten()


        # ----------------------------------------------------
        # KNN / SVM
        # ----------------------------------------------------

        else:

            background = (
                X_train
                .sample(
                    min(
                        50,
                        len(X_train)
                    ),
                    random_state=42
                )
            )

            def prediction_function(
                values
            ):

                values_df = pd.DataFrame(
                    values,
                    columns=FEATURES
                )

                return (
                    best_model
                    .predict_proba(
                        values_df
                    )[:, 1]
                )

            explainer = shap.Explainer(
                prediction_function,
                background
            )

            shap_result = explainer(
                patient_data
            )

            shap_values = (
                shap_result.values[0]
            )


        # ====================================================
        # CREATE EXPLANATION DATAFRAME
        # ====================================================

        shap_values = np.asarray(
            shap_values
        ).flatten()

        if len(shap_values) != len(FEATURES):

            raise ValueError(
                f"Unexpected number of SHAP values: "
                f"{len(shap_values)}"
            )

        explanation_df = pd.DataFrame({

            "Feature": FEATURES,

            "SHAP Value": shap_values

        })


        explanation_df[
            "Absolute SHAP"
        ] = (
            explanation_df[
                "SHAP Value"
            ].abs()
        )


        explanation_df = (
            explanation_df
            .sort_values(
                "Absolute SHAP",
                ascending=False
            )
            .reset_index(drop=True)
        )


        # ====================================================
        # SHAP TABLE
        # ====================================================

        st.subheader(
            "Feature Contributions"
        )

        st.dataframe(
            explanation_df[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ].round(4),
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SHAP BAR CHART
        # ====================================================

        chart_df = (
            explanation_df
            .sort_values(
                "SHAP Value"
            )
        )


        shap_fig = px.bar(

            chart_df,

            x="SHAP Value",

            y="Feature",

            orientation="h",

            title="Feature Contribution to Prediction"
        )


        shap_fig.add_vline(
            x=0,
            line_dash="dash"
        )


        shap_fig.update_layout(
            xaxis_title="SHAP Value",
            yaxis_title="Feature"
        )


        st.plotly_chart(
            shap_fig,
            use_container_width=True
        )


        # ====================================================
        # TOP FACTORS
        # ====================================================

        st.subheader(
            "Most Influential Features"
        )


        strongest = (
            explanation_df
            .head(3)
        )


        for _, row in strongest.iterrows():

            feature = row[
                "Feature"
            ]

            value = row[
                "SHAP Value"
            ]


            if value > 0:

                st.warning(
                    f"**{feature}** contributed "
                    f"toward a higher predicted "
                    f"diabetes probability "
                    f"(SHAP = {value:.4f})."
                )

            else:

                st.info(
                    f"**{feature}** contributed "
                    f"toward a lower predicted "
                    f"diabetes probability "
                    f"(SHAP = {value:.4f})."
                )


    except Exception as e:

        st.warning(
            "The prediction was generated, "
            "but the SHAP explanation could not "
            f"be displayed: {e}"
        )


    # ========================================================
    # GENERAL RECOMMENDATION
    # ========================================================

    st.divider()

    st.subheader(
        "💡 General Recommendation"
    )


    if risk_level == "High":

        st.warning(
            """
            The model estimates a relatively high
            probability of diabetes.

            This is not a medical diagnosis.
            A qualified healthcare professional
            should evaluate the patient's condition.
            """
        )


    elif risk_level == "Moderate":

        st.info(
            """
            The model estimates a moderate probability
            of diabetes.

            Consider discussing appropriate health
            testing and evaluation with a healthcare
            professional.
            """
        )


    else:

        st.success(
            """
            The model estimates a relatively low
            probability of diabetes.

            This result does not rule out medical
            conditions and should not replace
            professional medical advice.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Explainable AI Disease Prediction System | "
    "Educational Machine Learning Project"
)

st.caption(
    "This application is not a medical diagnostic system."
)