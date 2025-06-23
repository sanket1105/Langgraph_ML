#!/usr/bin/env python3

import os

import h2o
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from h20 import H2OMLAgentEnhanced


def access_existing_custom_models():
    """Access custom models from an existing H2O agent run and generate PDFs."""

    print("=== Accessing Existing Custom Models from H2O Agent ===\n")

    # 1. Create sample data (same as what was used in the original run)
    print("1. Creating sample data...")
    X, y = make_classification(
        n_samples=200, n_features=15, n_informative=8, n_redundant=4, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    # 2. Split data (same split as original run)
    print("2. Splitting data...")
    X = X_df  # Features
    y = y_series  # Target

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # 3. Create agent instance (without running it)
    print("3. Creating H2O ML Agent instance...")
    agent = H2OMLAgentEnhanced(model=None)

    # 4. Check if there are existing logs with custom models
    print("4. Checking for existing custom models...")

    # Look for existing PDFs to see what was generated
    logs_dir = "logs/"
    if os.path.exists(logs_dir):
        pdf_files = [f for f in os.listdir(logs_dir) if f.endswith(".pdf")]
        print(f"   Found {len(pdf_files)} PDF files in logs directory:")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(logs_dir, pdf_file)
            file_size = os.path.getsize(pdf_path)
            print(f"     📄 {pdf_file} ({file_size:,} bytes)")

    # 5. Demonstrate how to access custom models from agent response
    print("\n5. Demonstrating how to access custom models from agent response...")

    # This is how you would access custom models after running the agent:
    # actual_custom_models = agent.response.get("custom_models", None)

    # For demonstration, let's show what the structure should look like:
    print("   To access custom models after running the agent:")
    print("   ```python")
    print("   # After running ml_agent.invoke_agent(...)")
    print("   if ml_agent.response and 'custom_models' in ml_agent.response:")
    print("       actual_custom_models = ml_agent.response['custom_models']")
    print("       print(f'Found {len(actual_custom_models)} custom models:')")
    print("       for model_name in actual_custom_models.keys():")
    print("           print(f'  - {model_name}')")
    print("   ```")

    # 6. Show how to generate PDFs with existing custom models
    print("\n6. Demonstrating PDF generation with custom models...")

    # Initialize H2O for PDF generation
    h2o.init()
    test_h2o = h2o.H2OFrame(X_test)
    test_h2o["target"] = h2o.H2OFrame(y_test.to_frame())

    # Example of how to generate PDFs with custom models:
    print("   To generate PDFs with custom models:")
    print("   ```python")
    print("   # Generate combined PDF (H2O + Custom models)")
    print("   combined_pdf = agent.plot_combined_reliability_curves(")
    print("       leaderboard_df=leaderboard,")
    print("       test_h2o=test_h2o,")
    print("       target_variable='target',")
    print("       custom_models=actual_custom_models,")
    print("       logs_dir='logs/'")
    print("   )")
    print("   ")
    print("   # Generate custom models only PDF")
    print("   custom_pdf = agent.plot_custom_models_reliability_curves(")
    print("       custom_models=actual_custom_models,")
    print("       test_h2o=test_h2o,")
    print("       target_variable='target',")
    print("       logs_dir='logs/'")
    print("   )")
    print("   ```")

    # 7. Show what models should be included
    print("\n7. Expected custom models from H2O agent:")
    expected_models = [
        "Random Forest",
        "XGBoost",
        "CatBoost",
        "LightGBM",
        "Logistic Regression (L1)",
        "Logistic Regression (L2)",
    ]

    print("   The H2O agent should train these custom models:")
    for i, model_name in enumerate(expected_models, 1):
        print(f"     {i}. {model_name}")

    print("\n   These are the ACTUAL trained models with optimized parameters")
    print("   from Optuna hyperparameter tuning, not random models.")

    # 8. Summary
    print(f"\n=== SUMMARY ===")
    print(f"✓ The H2O agent stores actual trained custom models in its response")
    print(f"✓ Access them via: ml_agent.response['custom_models']")
    print(f"✓ These models have optimized hyperparameters from Optuna tuning")
    print(f"✓ Use them to generate PDFs with: plot_custom_models_reliability_curves()")
    print(f"✓ The PDFs will show the REAL performance of the trained models")

    # Cleanup
    h2o.cluster().shutdown()

    return agent


if __name__ == "__main__":
    access_existing_custom_models()
