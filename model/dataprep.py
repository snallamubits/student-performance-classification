"""
Data Preparation Module
=======================

Handles:
- Dataset loading via ucimlrepo
- Multi-level target creation
- Categorical feature encoding
- Numeric feature scaling
- Data leakage prevention
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ------------------------------------------------------------
# Helper: Map final grade to performance class
# ------------------------------------------------------------
def map_grade_to_class(g3):
    if g3 <= 9:
        return 0
    elif g3 <= 11:
        return 1
    elif g3 <= 13:
        return 2
    elif g3 <= 15:
        return 3
    else:
        return 4


# ------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------
def load_and_prepare_data():
    # Fetch dataset
    student_performance = fetch_ucirepo(id=320)

    X = student_performance.data.features
    y = student_performance.data.targets

    # Combine for processing
    df = pd.concat([X, y], axis=1)

    # Create multi-class target
    df["performance_level"] = df["G3"].apply(map_grade_to_class)

    # Remove grade columns to avoid leakage
    df.drop(columns=["G1", "G2", "G3"], inplace=True)

    # Separate features and target
    X = df.drop(columns=["performance_level"])
    y = df["performance_level"]

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing: encode categoricals, scale numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Apply transformations
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test
