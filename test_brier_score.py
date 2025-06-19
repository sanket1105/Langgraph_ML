"""
Test script for H2O AutoML Enhanced with Brier Score evaluation
"""

import os

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the enhanced H2O agent
from h20 import H2OMLAgentEnhanced


def create_test_data():
    """Create a synthetic binary classification dataset."""
    print("Creating synthetic binary classification dataset...")

    # Create imbalanced dataset (20% positive class)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.8, 0.2],  # 80% negative, 20% positive
        random_state=42,
        flip_y=0.1,  # Add some noise
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Positive class percentage: {df['target'].mean():.2%}")

    return df


def main():
    """Main test function."""
    print("=== H2O AutoML Enhanced with Brier Score Test ===\n")

    # Create test data
    df = create_test_data()

    # Split data into train/test/calibration
    X = df.drop("target", axis=1)
    y = df["target"]

    # First split: train+calib vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: train vs calib
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Calibration set size: {len(X_calib)}")
    print()

    # Initialize the LLM (you'll need to set your API key)
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("Please set your OpenAI API key and try again.")
        return

    # Initialize the enhanced H2O agent
    print("Initializing H2O ML Agent Enhanced...")
    ml_agent = H2OMLAgentEnhanced(
        model=llm,
        log=True,
        log_path="logs/",
        model_directory="models/",
        enable_optuna=True,
        optuna_n_trials=10,  # Reduced for faster testing
        optuna_timeout=60,  # 1 minute timeout
        human_in_the_loop=False,
        bypass_explain_code=True,
    )

    # Run the agent
    print("Starting H2O ML Agent Enhanced...")
    ml_agent.invoke_agent(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_calib=X_calib,
        y_calib=y_calib,
        user_instructions="Create an H2O AutoML model for binary classification. Focus on maximizing AUC and minimizing Brier score for well-calibrated probability estimates. Use the calibration set for model calibration assessment.",
    )

    # Get results
    print("\n=== Results ===")

    # Test metrics
    test_metrics = ml_agent.get_test_metrics()
    if test_metrics:
        print("Test Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Calibration metrics
    calib_metrics = ml_agent.get_calibration_metrics()
    if calib_metrics:
        print("\nCalibration Set Metrics:")
        for metric, value in calib_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Leaderboard
    leaderboard = ml_agent.get_leaderboard()
    if leaderboard is not None and not leaderboard.empty:
        print("\nH2O AutoML Leaderboard:")
        print(leaderboard.head())

    # Optimization results
    opt_results = ml_agent.get_optimization_results()
    if opt_results:
        print("\nOptuna Optimization Results:")
        print(f"  Best value: {opt_results.get('best_value', 'N/A')}")
        print(f"  Number of trials: {opt_results.get('n_trials', 'N/A')}")
        if "best_params" in opt_results:
            print("  Best parameters:")
            for param, value in opt_results["best_params"].items():
                print(f"    {param}: {value}")

    # Model path
    model_path = ml_agent.get_model_path()
    if model_path:
        print(f"\nModel saved to: {model_path}")

    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    main()
