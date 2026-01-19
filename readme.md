# Machine Learning Classification Models – Student Performance Dataset

## a. Problem Statement

The objective of this project is to design, implement, and evaluate multiple
machine learning classification models on a real-world educational dataset.
The goal is to predict student performance levels based on demographic,
social, and school-related attributes and to compare the performance of
different classification algorithms using standard evaluation metrics.

---

## b. Dataset Description

The dataset used in this project is the **Student Performance Dataset**
obtained from the UCI Machine Learning Repository.

The dataset represents student achievement in secondary education across
two Portuguese schools and was collected using school reports and
questionnaires. It includes academic, demographic, social, and school-related
features.

Two subjects are originally available in the dataset:
- Mathematics
- Portuguese Language

In this project, the dataset is accessed programmatically using the
`ucimlrepo` Python library (Dataset ID: 320), ensuring reproducibility and
data integrity.

### Dataset Characteristics
- Number of instances: More than 1000
- Number of features: More than 30
- Feature types: Numeric and categorical
- Target attribute: Final grade (G3)

### Target Engineering
The numeric final grade (G3) is converted into a **five-level classification**
problem:

| Class | Grade Range | Description |
|------|------------|-------------|
| 0 | 0–9 | Very Poor |
| 1 | 10–11 | Poor |
| 2 | 12–13 | Satisfactory |
| 3 | 14–15 | Good |
| 4 | 16–20 | Excellent |

To avoid data leakage, intermediate grades (G1 and G2) are excluded from the
feature set, making the prediction task more realistic and meaningful.

---

## c. Models Used

The following six classification models were implemented using the same
dataset and train–test split:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN) Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Gaussian Naive Bayes was selected as the feature set consists of continuous
numerical values after preprocessing.

---

## d. Evaluation Metrics

Each model was evaluated using the following metrics:

- Accuracy  
- AUC Score (One-vs-Rest, Macro Average)  
- Precision (Macro Average)  
- Recall (Macro Average)  
- F1 Score (Macro Average)  
- Matthews Correlation Coefficient (MCC)  

These metrics provide a balanced evaluation of model performance,
especially for multi-class classification problems.

---

## e. Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|---------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.71 | 0.79 | 0.68 | 0.66 | 0.67 | 0.58 |
| Decision Tree | 0.69 | 0.74 | 0.65 | 0.64 | 0.64 | 0.54 |
| KNN | 0.70 | 0.77 | 0.66 | 0.65 | 0.65 | 0.56 |
| Naive Bayes | 0.67 | 0.73 | 0.63 | 0.61 | 0.62 | 0.51 |
| Random Forest (Ensemble) | 0.75 | 0.83 | 0.72 | 0.71 | 0.71 | 0.62 |
| XGBoost (Ensemble) | 0.78 | 0.86 | 0.75 | 0.74 | 0.74 | 0.66 |

---

## f. Observations on Model Performance

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Performs well as a baseline model but is limited in capturing complex non-linear relationships. |
| Decision Tree | Captures feature interactions effectively but shows signs of overfitting. |
| KNN | Provides moderate performance and is sensitive to feature scaling and choice of K. |
| Naive Bayes | Computationally efficient but constrained by the independence assumption. |
| Random Forest (Ensemble) | Demonstrates strong and stable performance by reducing variance through ensemble learning. |
| XGBoost (Ensemble) | Achieves the best overall performance due to boosting, regularization, and efficient handling of complex patterns. |

---

## g. Repository Structure
project-folder/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── README.md
│
└── model/
├── init.py
├── datapreparation.py
└── modelsmetrics.py

## h. Execution Instructions

### Local Execution

pip install -r requirements.txt
python app.py
streamlit run streamlit_app.py


i. Deployment

The Streamlit application is deployed using Streamlit Community Cloud
and provides an interactive interface to run all models and view the
comparison results.


j. Conclusion

This project demonstrates a comparative analysis of multiple classification
models on an educational dataset. Ensemble methods, particularly XGBoost,
outperform simpler models due to their ability to capture complex patterns
and reduce bias and variance.
