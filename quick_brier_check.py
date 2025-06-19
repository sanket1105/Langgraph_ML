"""
Quick verification script to check if H2O method calculates Brier score
"""

import os

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from h20 import H2OMLAgentEnhanced


def quick_brier_check():
    """Quick check to verify Brier score calculation."""

    print("=== Quick Brier Score Check ===\n")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable first")
        return False

    # Create simple test data
    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    # Split data
    X = df.drop(columns=["target"])
    y = df["target"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}, Calib={len(X_calib)}")

    # Initialize agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    ml_agent = H2OMLAgentEnhanced(
        model=llm,
        log=True,
        log_path="logs/",
        model_directory="models/",
        enable_optuna=False,  # Disable for speed
        human_in_the_loop=False,
        bypass_explain_code=True,
    )

    # Run with explicit Brier score instructions
    print("Running H2O agent...")
    ml_agent.invoke_agent(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_calib=X_calib,
        y_calib=y_calib,
        user_instructions="Calculate Brier score for both test and calibration sets. This is essential for model calibration assessment.",
    )

    # Check results
    test_metrics = ml_agent.get_test_metrics()
    calib_metrics = ml_agent.get_calibration_metrics()

    print("\n=== Results ===")

    success = True

    # Check test metrics
    if test_metrics and "brier_score" in test_metrics:
        print(f"✅ Test Brier Score: {test_metrics['brier_score']:.6f}")
    else:
        print("❌ Test Brier Score: NOT FOUND")
        success = False

    # Check calibration metrics
    if calib_metrics and "brier_score" in calib_metrics:
        print(f"✅ Calibration Brier Score: {calib_metrics['brier_score']:.6f}")
    else:
        print("❌ Calibration Brier Score: NOT FOUND")
        success = False

    # Check generated code
    code = ml_agent.get_h2o_train_function()
    if code and "brier_score" in code:
        print("✅ Generated code includes Brier score calculation")
    else:
        print("❌ Generated code missing Brier score calculation")
        success = False

    # Summary
    if success:
        print("\n🎉 SUCCESS: H2O method is calculating Brier scores correctly!")
        return True
    else:
        print("\n❌ ISSUE: Brier score calculation needs attention")
        return False


if __name__ == "__main__":
    quick_brier_check()
