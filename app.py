# ------------------------------------------------------------
# Main Streamlit Application 

# Runs multiple classification models on the same dataset  and displays evaluation metrics along with qualitative observations on model performance.
# ------------------------------------------------------------

import streamlit as st
import pandas as pd

from model.datapreparation import load_and_prepare_data
from model.modelsmetrics import run_all_models


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

    # --------------------------------------------------------
    # Results Table
    # --------------------------------------------------------
    st.subheader("Model Comparison Table")
    st.dataframe(results_df.set_index("Model").round(4))

    # --------------------------------------------------------
    # Observations Section
    # --------------------------------------------------------
    st.subheader("Observations on Model Performance")

    observations = {
        "Logistic Regression":            "Served as a reliable baseline model with consistent performance, but its linear nature limited its ability to capture complex relationships in the data.",

        "Decision Tree":            "Was effective at learning feature interactions, though its performance variability suggested a tendency to overfit.",

        "KNN":            "Delivered reasonable results but remained sensitive to feature scaling and neighborhood size, affecting prediction stability.",

        "Naive Bayes":            "Executed efficiently with acceptable performance; however, the independence assumption reduced accuracy on correlated features.",

        "Random Forest":            "Showed strong and stable performance by combining multiple trees, which helped improve generalization and reduce variance.",

        "XGBoost":            "Achieved the best overall performance due to boosting, regularization, and its ability to model complex patterns efficiently."
    }

    obs_df = pd.DataFrame(
        list(observations.items()),
        columns=["ML Model", "Observation"]
    )

    st.table(obs_df)
