# app.py
# Requirements:
# pip install streamlit pandas scikit-learn joblib

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# ---------- Utility: load artifacts ----------
@st.cache_data(show_spinner=False)
def load_artifacts():
    artifacts = {}
    base = Path(".")
    try:
        artifacts["scaler"] = joblib.load(base / "scaler.joblib")
    except Exception as e:
        artifacts["scaler"] = None
    try:
        artifacts["cat_cols"] = joblib.load(base / "categorical_dummies_columns.joblib")
    except Exception as e:
        artifacts["cat_cols"] = None
    try:
        artifacts["clf"] = joblib.load(base / "student_perf_classifier_rf.joblib")
    except Exception as e:
        artifacts["clf"] = None
    try:
        artifacts["reg"] = joblib.load(base / "student_perf_regressor_rf.joblib")
    except Exception as e:
        artifacts["reg"] = None
    return artifacts

artifacts = load_artifacts()
missing = [k for k, v in artifacts.items() if v is None]
if missing:
    st.warning(
        "Some model artifacts are missing: " + ", ".join(missing) +
        ".\nMake sure the files scaler.joblib, categorical_dummies_columns.joblib, "
        "student_perf_classifier_rf.joblib and student_perf_regressor_rf.joblib are in the same folder as this app."
    )

# ---------- Constants (must match training) ----------
NUMERIC_FEATURES = [
    "age",
    "attendance_percentage",
    "math_marks",
    "science_marks",
    "english_marks",
    "study_hours_per_week",
    "homework_completion_rate",
]

GENDER_OPTIONS = ["Male", "Female"]
PARENT_EDU_OPTIONS = ["Primary", "Secondary", "Graduate", "Post-Graduate"]

# ---------- Header ----------
st.title("🎯 Student Performance Predictor")
st.write("Enter student details (single sample) or upload a CSV to predict performance category and final score.")

# ---------- Sidebar: Model status ----------
with st.sidebar:
    st.header("Model artifacts")
    for name, val in artifacts.items():
        status = "✅ Loaded" if val is not None else "❌ Missing"
        st.write(f"- {name}: {status}")
    st.markdown("---")
    st.write("CSV input must contain columns:")
    st.write(", ".join(
        NUMERIC_FEATURES + ["gender", "parent_education_level"]
    ))

# ---------- Input form ----------
st.subheader("Single student input")
with st.form("single_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=25, value=16)
        attendance_percentage = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=92.5, format="%.2f")
        study_hours_per_week = st.number_input("Study hours / week", min_value=0, max_value=100, value=8)
    with col2:
        math_marks = st.number_input("Math marks (0-100)", min_value=0, max_value=100, value=78)
        science_marks = st.number_input("Science marks (0-100)", min_value=0, max_value=100, value=85)
        english_marks = st.number_input("English marks (0-100)", min_value=0, max_value=100, value=72)
    with col3:
        homework_completion_rate = st.number_input("Homework completion (%)", min_value=0.0, max_value=100.0, value=90.0, format="%.2f")
        gender = st.selectbox("Gender", options=GENDER_OPTIONS)
        parent_education_level = st.selectbox("Parent education level", options=PARENT_EDU_OPTIONS)

    submitted = st.form_submit_button("Predict single")

# ---------- Helper: Prepare dataframe as in training ----------
def prepare_features(df_input, scaler, cat_cols):
    """
    df_input: DataFrame with columns NUMERIC_FEATURES + ['gender', 'parent_education_level']
    scaler: fitted scaler (or None)
    cat_cols: list of categorical dummy column names used in training (or None)
    """
    df = df_input.copy()
    # Ensure numeric columns exist and are numeric
    for c in NUMERIC_FEATURES:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Scale numeric
    if scaler is not None:
        scaled = scaler.transform(df[NUMERIC_FEATURES])
        df_num = pd.DataFrame(scaled, columns=NUMERIC_FEATURES, index=df.index)
    else:
        # If scaler missing, pass raw numeric (with a warning shown elsewhere)
        df_num = df[NUMERIC_FEATURES].astype(float)

    # One-hot encode categoricals (same approach as training: pandas.get_dummies)
    cat_df = pd.get_dummies(df[[ "gender", "parent_education_level" ]].astype(str), drop_first=True)

    # Align categorical columns to training
    if cat_cols is not None:
        for c in cat_cols:
            if c not in cat_df.columns:
                cat_df[c] = 0
        cat_df = cat_df.reindex(columns=cat_cols, fill_value=0)
    # else: keep whatever columns we have

    X_prepared = pd.concat([df_num.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    return X_prepared

# ---------- Single-sample prediction ----------
if submitted:
    sample = pd.DataFrame([{
        "age": age,
        "attendance_percentage": attendance_percentage,
        "math_marks": math_marks,
        "science_marks": science_marks,
        "english_marks": english_marks,
        "study_hours_per_week": study_hours_per_week,
        "homework_completion_rate": homework_completion_rate,
        "gender": gender,
        "parent_education_level": parent_education_level
    }])

    Xs = prepare_features(sample, artifacts.get("scaler"), artifacts.get("cat_cols"))

    st.markdown("**Prepared features (first rows)**")
    st.dataframe(Xs.T, use_container_width=True)

    # Classification
    if artifacts.get("clf") is not None:
        try:
            pred_class = artifacts["clf"].predict(Xs)[0]
            proba = artifacts["clf"].predict_proba(Xs)[0]
            classes = artifacts["clf"].classes_
            proba_df = pd.DataFrame({"class": classes, "probability": proba}).sort_values("probability", ascending=False)
            st.success(f"Predicted performance category: **{pred_class}**")
            st.table(proba_df)
        except Exception as e:
            st.error(f"Error during classification prediction: {e}")
    else:
        st.error("Classification model not loaded.")

    # Regression
    if artifacts.get("reg") is not None:
        try:
            pred_score = artifacts["reg"].predict(Xs)[0]
            st.info(f"Predicted final score (estimate): **{pred_score:.2f}**")
            # map score to same buckets as training
            if pred_score >= 85:
                mapped = "High"
            elif pred_score >= 60:
                mapped = "Medium"
            else:
                mapped = "Low"
            st.write(f"Mapped performance category from predicted score: **{mapped}**")
        except Exception as e:
            st.error(f"Error during regression prediction: {e}")
    else:
        st.error("Regression model not loaded.")

# ---------- Batch CSV upload ----------
st.markdown("---")
st.subheader("Batch predictions (CSV upload)")

uploaded_file = st.file_uploader("Upload CSV containing columns: " + ", ".join(NUMERIC_FEATURES + ["gender", "parent_education_level"]), type=["csv"])
if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        df_input = None

    if df_input is not None:
        st.write("Preview of uploaded file:")
        st.dataframe(df_input.head())

        X_batch = prepare_features(df_input, artifacts.get("scaler"), artifacts.get("cat_cols"))

        results = pd.DataFrame(index=X_batch.index)
        if artifacts.get("clf") is not None:
            try:
                results["predicted_category"] = artifacts["clf"].predict(X_batch)
            except Exception as e:
                st.error(f"Batch classification error: {e}")
        if artifacts.get("reg") is not None:
            try:
                results["predicted_final_score"] = artifacts["reg"].predict(X_batch).round(2)
            except Exception as e:
                st.error(f"Batch regression error: {e}")

        out = pd.concat([df_input.reset_index(drop=True), results.reset_index(drop=True)], axis=1)
        st.write("Predictions preview:")
        st.dataframe(out.head())

        # Allow download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Upload a CSV to run batch predictions.")

# ---------- Footer / tips ----------
st.markdown("---")
st.write("Tips:")
st.write("- Be sure the training artifacts are in the app folder: `scaler.joblib`, `categorical_dummies_columns.joblib`, `student_perf_classifier_rf.joblib`, `student_perf_regressor_rf.joblib`.")
st.write("- If you retrain with different categorical levels, re-generate `categorical_dummies_columns.joblib` and replace it here.")
st.write("- I can update the app to include EDA plots, shap/feature importance, or a nicer layout — tell me which one you want!")
