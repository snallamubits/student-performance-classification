# ------------------------------------------------------------
# Main Streamlit Application 

# Runs multiple classification models on the same dataset  and displays evaluation metrics along with qualitative observations on model performance.
# ------------------------------------------------------------

import streamlit as st
import pandas as pd

from model.dataprep import load_and_prepare_data
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
        "Logistic Regression":        "This was a reliable baseline model with consistent performance, but its linear nature limits its ability to capture more complex relationships present in the data.",

        "Decision Tree":        "Was able to model feature interactions effectively; however, its performance fluctuated, indicating a tendency to overfit the training data.",

        "KNN":        "This model produced reasonable results but showed sensitivity to feature scaling and neighborhood size, which affected its stability across different samples.",

        "Naive Bayes":        "Executed very quickly and provided acceptable results, though its strong independence assumption reduced accuracy on correlated features.",

        "Random Forest":        "Demonstrated strong and consistent performance by aggregating multiple decision trees, which helped reduce variance and improve generalization.",

        "XGBoost":        "Delivered the best overall results, benefiting from boosting, regularization, and its ability to learn complex patterns efficiently."
    }

    obs_df = pd.DataFrame(
        list(observations.items()),
        columns=["ML Model", "Observation"]
    )

    st.table(obs_df)

