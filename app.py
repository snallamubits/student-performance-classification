"""
Main execution file.

Runs all classification models on the same dataset
and prints a comparison table with evaluation metrics.
"""

from model.dataprep import load_and_prepare_data
from model.metrics import run_all_models


def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    results_df = run_all_models(X_train, X_test, y_train, y_test)

    print("\nModel Comparison Table:\n")
    print(results_df.set_index("Model").round(4))


if __name__ == "__main__":
    main()