"""
Main execution file.

Runs all classification models on the same dataset
and prints a comparison table with evaluation metrics.
"""

import streamlit as st
import pandas as pd

from model.datapre import load_and_prepare_data
from model.metrics import run_all_models

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Classification",
    layout="wide"
)

# ------------------------------------------------------------
# UI Header
# ------------------------------------------------------------
st.title("Student Performance Classification Models")
st.write(
    "This application compares multiple machine learning classification "
    "models using the Student Performance dataset."
)

# ------------------------------------------------------------
# Run Models Button
# ------------------------------------------------------------
if st.button("Run Models"):
    with st.spinner("Training models and evaluating performance..."):
        X_train, X_test, y_train, y_test = load_and_prepare_data()
        results_df = run_all_models(X_train, X_test, y_train, y_test)

st.success("Model execution completed!")

st.subheader("Model Comparison Table")
st.dataframe(results_df.set_index("Model").round(4))


st.subheader("Observations on Model Performance")

# ------------------------------------------------------------
# Observations table
# ------------------------------------------------------------
observations = {
        "Logistic Regression": "Performs well as a baseline model but is limited in capturing complex non-linear relationships.",
        "Decision Tree": "Captures feature interactions effectively but may overfit the training data.",
        "KNN": "Shows moderate performance and is sensitive to feature scaling and the choice of K.",
        "Naive Bayes":  "Computationally efficient but constrained by the independence assumption among features.",
        "Random Forest": "Provides strong and stable performance by reducing variance through ensemble learning.",
        "XGBoost": "Achieves the best overall performance due to boosting, regularization, and efficient learning."
    }

obs_df = pd.DataFrame(
        list(observations.items()),
        columns=["ML Model", "Observation"]
    )

st.table(obs_df)

