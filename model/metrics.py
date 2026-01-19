"""
Model training and evaluation module.

Implements:
1. Logistic Regression
2. Decision Tree
3. KNN
4. Naive Bayes
5. Random Forest
6. XGBoost

Evaluates:
Accuracy, AUC, Precision, Recall, F1, MCC
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(
            y_test, y_prob,
            multi_class="ovr",
            average="macro"
        )
    except:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }


def run_all_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        # Classify the given sample based on the 5 nearest training samples.
        "KNN": KNeighborsClassifier(n_neighbors=5),
        # Naive Bayes (Gaussian) chosen as features are continuous
        # Gaussian Naive Bayes was selected because the feature set contains continuous numerical values after preprocessing, making it more suitable than Multinomial Naive Bayes, which is typically used for count-based or discrete data.
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)

    return pd.DataFrame(results)
