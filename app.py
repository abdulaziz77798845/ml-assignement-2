import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Comparison", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification – Model Evaluation")
st.write("Upload a labelled test CSV, select a trained classifier, and evaluate its predictions. Models are loaded from disk; the app does not retrain them.")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]
TARGET = "target"

uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"])
model_name = st.selectbox("Select classification model", list(MODEL_FILES.keys()))

if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        missing = [c for c in FEATURES + [TARGET] if c not in test_df.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif test_df.empty:
            st.warning("The uploaded CSV contains no rows.")
        else:
            X_test = test_df[FEATURES].apply(pd.to_numeric, errors="coerce")
            y_test = pd.to_numeric(test_df[TARGET], errors="coerce")
            if X_test.isnull().any().any() or y_test.isnull().any():
                st.error("Input features and target must contain valid numeric values without missing entries.")
            elif not set(y_test.unique()).issubset({0, 1}):
                st.error("Target must contain 0 (malignant) or 1 (benign).")
            else:
                model_path = MODEL_FILES[model_name]
                if not os.path.exists(model_path):
                    st.error(f"Saved model not found: {model_path}")
                else:
                    model = joblib.load(model_path)
                    y_pred = model.predict(X_test)
                    y_score = model.predict_proba(X_test)[:, 1]
                    metrics = {
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "AUC": roc_auc_score(y_test, y_score),
                        "Precision": precision_score(y_test, y_pred, zero_division=0),
                        "Recall": recall_score(y_test, y_pred, zero_division=0),
                        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
                        "MCC": matthews_corrcoef(y_test, y_pred),
                    }
                    st.subheader(f"Evaluation results: {model_name}")
                    cols = st.columns(3)
                    for col, (label, value) in zip(cols + st.columns(3), metrics.items()):
                        col.metric(label, f"{value:.4f}")
                    st.subheader("Confusion matrix")
                    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                    st.dataframe(pd.DataFrame(cm, index=["Actual malignant (0)", "Actual benign (1)"],
                                              columns=["Predicted malignant (0)", "Predicted benign (1)"]))
                    st.subheader("Classification report")
                    st.text(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"], zero_division=0))
                    with st.expander("Preview uploaded test data"):
                        st.dataframe(test_df.head(20))
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload the included test_data.csv to view evaluation metrics.")
    st.caption("Target encoding: 0 = malignant; 1 = benign. The CSV must include all 30 feature columns and target.")
