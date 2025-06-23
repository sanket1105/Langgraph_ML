#!/usr/bin/env python3

import os

import pandas as pd
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from h20 import H2OMLAgentEnhanced


def demo_actual_custom_models():
    """Demonstrate how to access the actual trained custom models from H2O agent."""

    print("=== Demonstrating Actual Custom Models from H2O Agent ===\n")

    # 1. Create sample data
    print("1. Creating sample data...")
    X, y = make_classification(
        n_samples=200, n_features=15, n_informative=8, n_redundant=4, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")

    # 2. Split the data into train/test/calibration sets
    print("2. Splitting data...")
    X = X_df.drop(columns=["target"])
    y = y_series

    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: separate calibration set from remaining data (25% of temp = 20% of total)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"   Train set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")
    print(f"   Calibration set size: {len(X_calib)}")

    # 3. Set up directories
    LOG_PATH = "logs/"
    MODEL_PATH = "models/"
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    # 4. Initialize the H2O ML Agent
    print("3. Initializing H2O ML Agent...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    ml_agent = H2OMLAgentEnhanced(
        model=llm,
        log=True,
        log_path=LOG_PATH,
        model_directory=MODEL_PATH,
        n_samples=30,
        file_name="h2o_automl_enhanced.py",
        function_name="h2o_automl_enhanced",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        enable_mlflow=False,
        enable_optuna=True,
        optuna_n_trials=10,  # Reduced for faster demo
        optuna_timeout=120,  # Reduced for faster demo
    )

    # 5. Run the agent to train actual models
    print("4. Running H2O ML Agent to train actual models...")
    result = ml_agent.invoke_agent(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_calib=X_calib,
        y_calib=y_calib,
        user_instructions="""
        Please create an H2O AutoML model for binary classification.
        Focus on maximizing AUC score while maintaining good precision.
        Use the calibration set for model calibration and threshold optimization.
        Optimize hyperparameters using Optuna for best performance.
        """,
        max_retries=2,
    )

    # 6. Access the actual trained custom models from the agent
    print("5. Accessing actual trained custom models from agent response...")

    if ml_agent.response and "custom_models" in ml_agent.response:
        actual_custom_models = ml_agent.response["custom_models"]
        print(f"   ✓ Found {len(actual_custom_models)} actual trained custom models:")

        for model_name in actual_custom_models.keys():
            print(f"     - {model_name}")

        # 7. Generate PDFs with the actual trained models
        print("6. Generating PDFs with actual trained models...")

        # Get the leaderboard to create test H2O frame
        leaderboard = ml_agent.get_leaderboard()
        if leaderboard is not None:
            # Create test H2O frame for PDF generation
            import h2o

            test_h2o = h2o.H2OFrame(X_test)
            test_h2o["target"] = h2o.H2OFrame(y_test.to_frame())

            # Generate custom models only PDF with actual trained models
            custom_pdf_path = ml_agent.plot_custom_models_reliability_curves(
                custom_models=actual_custom_models,
                test_h2o=test_h2o,
                target_variable="target",
                logs_dir=LOG_PATH,
            )
            print(f"   ✓ Custom models PDF generated: {custom_pdf_path}")

            # Show model details
            print("\n7. Model details:")
            for model_name, model in actual_custom_models.items():
                try:
                    # Get model parameters
                    params = model.get_params()
                    print(f"   {model_name}:")
                    print(f"     - Type: {type(model).__name__}")
                    print(f"     - Parameters: {len(params)} parameters")
                    # Show a few key parameters
                    key_params = {
                        k: v
                        for k, v in params.items()
                        if k
                        in [
                            "n_estimators",
                            "max_depth",
                            "learning_rate",
                            "C",
                            "penalty",
                        ]
                    }
                    if key_params:
                        print(f"     - Key params: {key_params}")
                except Exception as e:
                    print(f"   {model_name}: Error getting details - {e}")

            # Cleanup
            h2o.cluster().shutdown()

        else:
            print("   ❌ No leaderboard found in agent response")
    else:
        print("   ❌ No custom models found in agent response")
        print("   This might happen if:")
        print("   - The agent failed to train custom models")
        print("   - The response structure is different than expected")
        print("   - The agent didn't complete successfully")

    # 8. Summary
    print(f"\n=== SUMMARY ===")
    print(f"✓ Agent completed: {'Yes' if ml_agent.response else 'No'}")
    if ml_agent.response and "custom_models" in ml_agent.response:
        print(
            f"✓ Actual custom models found: {len(ml_agent.response['custom_models'])}"
        )
        print(f"✓ These are the REAL trained models with optimized parameters")
        print(f"✓ PDFs generated using actual trained models, not random ones")
    else:
        print(f"❌ No actual custom models found in agent response")

    return ml_agent


if __name__ == "__main__":
    demo_actual_custom_models()
