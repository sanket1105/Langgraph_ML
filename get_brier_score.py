"""
Script to get Brier score from the existing H2O session
"""

import h2o
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_brier_score_from_session():
    """Get Brier score from the existing H2O session."""

    print("=== Getting Brier Score from H2O Session ===\n")

    # Connect to existing H2O session
    try:
        h2o.connect()
        print("✅ Connected to existing H2O session")
    except Exception as e:
        print(f"❌ Could not connect to H2O session: {e}")
        return

    # Get the model from the session
    try:
        model = h2o.get_model("StackedEnsemble_AllModels_1_AutoML_13_20250618_205602")
        print("✅ Model retrieved from session")
    except Exception as e:
        print(f"❌ Could not get model from session: {e}")
        print("Available models:")
        try:
            models = h2o.ls()
            print(models)
        except:
            print("Could not list models")
        return

    # Recreate the test data
    print("\nRecreating test data...")

    # Create the same dataset as in the notebook
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # Split the data the same way
    X = df.drop(columns=["target"])
    y = df["target"]

    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: separate calibration set from remaining data (25% of temp = 20% of total)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Test set size: {len(X_test)}")
    print(f"Calibration set size: {len(X_calib)}")

    # Convert test data to H2O frame
    test_df = pd.concat([X_test, y_test], axis=1)
    test_h2o = h2o.H2OFrame(test_df)

    # Convert target to categorical
    test_h2o["target"] = test_h2o["target"].asfactor()

    # Get predictions
    print("\nGetting predictions...")
    test_pred = model.predict(test_h2o)

    # Extract probabilities and actual values
    test_probs = (
        test_pred["p1"].as_data_frame().values.flatten()
    )  # Probability of positive class
    test_actual = test_h2o["target"].as_data_frame().values.flatten()

    # Convert categorical to numeric
    if test_actual.dtype == "object":
        # Find the first unique value and use it as reference
        unique_values = np.unique(test_actual)
        test_actual_numeric = (test_actual == unique_values[0]).astype(int)
        # If the first value is 1, we need to flip (since we want 1 to be positive class)
        if unique_values[0] == 1:
            test_actual_numeric = 1 - test_actual_numeric
    else:
        test_actual_numeric = test_actual.astype(int)

    # Calculate Brier score
    brier_score = np.mean((test_probs - test_actual_numeric) ** 2)

    print(f"\n📊 Brier Score Results:")
    print(f"Brier Score: {brier_score:.6f}")
    print(f"Number of samples: {len(test_probs)}")
    print(f"Positive class percentage: {test_actual_numeric.mean():.2%}")

    # Show some example predictions
    print(f"\n📋 Sample Predictions (first 10):")
    print("Actual | Predicted Prob | Squared Error")
    print("-" * 35)
    for i in range(min(10, len(test_probs))):
        actual = test_actual_numeric[i]
        prob = test_probs[i]
        squared_error = (prob - actual) ** 2
        print(f"  {actual}   |     {prob:.4f}     |    {squared_error:.6f}")

    # Calculate additional metrics for comparison
    from sklearn.metrics import brier_score_loss, roc_auc_score

    sklearn_brier = brier_score_loss(test_actual_numeric, test_probs)
    auc_score = roc_auc_score(test_actual_numeric, test_probs)

    print(f"\n🔍 Verification:")
    print(f"Our Brier Score: {brier_score:.6f}")
    print(f"sklearn Brier Score: {sklearn_brier:.6f}")
    print(f"AUC Score: {auc_score:.6f}")

    # Check if they match
    if abs(brier_score - sklearn_brier) < 1e-10:
        print("✅ Brier scores match perfectly!")
    else:
        print(f"⚠️  Brier scores differ by {abs(brier_score - sklearn_brier):.10f}")

    # Interpret the Brier score
    print(f"\n📈 Brier Score Interpretation:")
    if brier_score < 0.05:
        print("Excellent calibration")
    elif brier_score < 0.10:
        print("Good calibration")
    elif brier_score < 0.15:
        print("Fair calibration")
    elif brier_score < 0.20:
        print("Poor calibration")
    else:
        print("Very poor calibration")

    # Calculate for calibration set too
    print(f"\n🎯 Calibration Set Brier Score:")
    calib_df = pd.concat([X_calib, y_calib], axis=1)
    calib_h2o = h2o.H2OFrame(calib_df)
    calib_h2o["target"] = calib_h2o["target"].asfactor()

    calib_pred = model.predict(calib_h2o)
    calib_probs = calib_pred["p1"].as_data_frame().values.flatten()
    calib_actual = calib_h2o["target"].as_data_frame().values.flatten()

    # Convert categorical to numeric
    if calib_actual.dtype == "object":
        unique_values = np.unique(calib_actual)
        calib_actual_numeric = (calib_actual == unique_values[0]).astype(int)
        if unique_values[0] == 1:
            calib_actual_numeric = 1 - calib_actual_numeric
    else:
        calib_actual_numeric = calib_actual.astype(int)

    calib_brier = np.mean((calib_probs - calib_actual_numeric) ** 2)
    print(f"Calibration Brier Score: {calib_brier:.6f}")

    print(f"\n✅ Brier score calculation completed!")

    # Also check what metrics are available from the model performance
    print(f"\n🔍 Model Performance Metrics:")
    test_perf = model.model_performance(test_h2o)
    print(f"Available metrics: {dir(test_perf)}")

    # Try to get AUC
    try:
        auc = test_perf.auc()
        print(f"AUC: {auc}")
    except:
        print("Could not get AUC")

    # Try to get logloss
    try:
        logloss = test_perf.logloss()
        print(f"LogLoss: {logloss}")
    except:
        print("Could not get LogLoss")


if __name__ == "__main__":
    get_brier_score_from_session()
