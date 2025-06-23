"""
Comprehensive guide to ensure H2O method calculates and returns Brier score
"""

import os

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from h20 import H2OMLAgentEnhanced


def ensure_brier_score_calculation():
    """
    Comprehensive example showing how to ensure H2O method calculates and returns Brier score.
    """

    print("=== Ensuring H2O Method Calculates Brier Score ===\n")

    # 1. Set up environment
    print("1. Setting up environment...")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set!")
        print("Please set your OpenAI API key and try again.")
        return

    # Create directories
    LOG_PATH = "logs/"
    MODEL_PATH = "models/"
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    # 2. Create test data
    print("2. Creating test data...")

    # Create synthetic binary classification dataset
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

    # Split data
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

    print(f"   Train set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")
    print(f"   Calibration set size: {len(X_calib)}")

    # 3. Initialize the H2O ML Agent with explicit Brier score instructions
    print("3. Initializing H2O ML Agent Enhanced...")

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
        optuna_n_trials=10,  # Reduced for faster testing
        optuna_timeout=120,
    )

    # 4. Run the agent with explicit Brier score instructions
    print("4. Running H2O ML Agent with Brier score focus...")

    ml_agent.invoke_agent(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_calib=X_calib,
        y_calib=y_calib,
        user_instructions="""
        Create an H2O AutoML model for binary classification with the following requirements:
        
        1. MUST calculate and return Brier score for both test and calibration sets
        2. Focus on maximizing AUC score while maintaining good calibration
        3. Use the calibration set for model calibration assessment
        4. Ensure the Brier score is properly stored in test_metrics and calibration_metrics
        5. The Brier score should be calculated as: mean((predicted_probability - actual_outcome)²)
        6. For binary classification, use the probability of the positive class (p1)
        7. Convert categorical targets to numeric (0/1) before Brier score calculation
        
        The Brier score is crucial for this analysis - make sure it's calculated correctly!
        """,
        max_retries=3,
    )

    # 5. Verify and display results
    print("5. Verifying Brier score calculation...")

    # Check if we have a response
    if not ml_agent.response:
        print("❌ No response from agent")
        return

    print(f"   Response keys: {list(ml_agent.response.keys())}")

    # 6. Get test metrics (should include Brier score)
    print("6. Retrieving test metrics...")
    test_metrics = ml_agent.get_test_metrics()

    if test_metrics:
        print("   ✅ Test metrics retrieved successfully")
        print("   Test Set Metrics:")
        for metric, value in test_metrics.items():
            print(f"     {metric}: {value:.6f}")

        # Check specifically for Brier score
        if "brier_score" in test_metrics:
            print(
                f"   ✅ Brier score found in test metrics: {test_metrics['brier_score']:.6f}"
            )
        else:
            print("   ❌ Brier score NOT found in test metrics")
    else:
        print("   ❌ No test metrics available")

    # 7. Get calibration metrics (should include Brier score)
    print("7. Retrieving calibration metrics...")
    calib_metrics = ml_agent.get_calibration_metrics()

    if calib_metrics:
        print("   ✅ Calibration metrics retrieved successfully")
        print("   Calibration Set Metrics:")
        for metric, value in calib_metrics.items():
            print(f"     {metric}: {value:.6f}")

        # Check specifically for Brier score
        if "brier_score" in calib_metrics:
            print(
                f"   ✅ Brier score found in calibration metrics: {calib_metrics['brier_score']:.6f}"
            )
        else:
            print("   ❌ Brier score NOT found in calibration metrics")
    else:
        print("   ❌ No calibration metrics available")

    # 8. Verify the generated code includes Brier score calculation
    print("8. Verifying generated code...")
    generated_code = ml_agent.get_h2o_train_function()

    if generated_code:
        # Check if Brier score calculation is in the code
        brier_keywords = [
            "brier_score",
            "np.mean((test_probs - test_actual) ** 2)",
            "np.mean((calib_probs - calib_actual) ** 2)",
            "test_metrics['brier_score']",
            "calib_metrics['brier_score']",
        ]

        code_has_brier = all(keyword in generated_code for keyword in brier_keywords)

        if code_has_brier:
            print("   ✅ Generated code includes Brier score calculation")
        else:
            print("   ❌ Generated code missing Brier score calculation")
            print(
                "   Missing keywords:",
                [kw for kw in brier_keywords if kw not in generated_code],
            )
    else:
        print("   ❌ No generated code available")

    # 9. Manual verification using the saved model
    print("9. Manual verification of Brier score...")

    try:
        import h2o

        # Connect to existing H2O session
        h2o.connect()

        # Get the model
        best_model_id = ml_agent.get_best_model_id()
        if best_model_id:
            model = h2o.get_model(best_model_id)

            # Recreate test data for verification
            test_df = pd.concat([X_test, y_test], axis=1)
            test_h2o = h2o.H2OFrame(test_df)
            test_h2o["target"] = test_h2o["target"].asfactor()

            # Get predictions
            test_pred = model.predict(test_h2o)
            test_probs = test_pred["p1"].as_data_frame().values.flatten()
            test_actual = test_h2o["target"].as_data_frame().values.flatten()

            # Convert categorical to numeric
            unique_values = np.unique(test_actual)
            test_actual_numeric = (test_actual == unique_values[0]).astype(int)
            if unique_values[0] == 1:
                test_actual_numeric = 1 - test_actual_numeric

            # Calculate Brier score manually
            manual_brier = np.mean((test_probs - test_actual_numeric) ** 2)

            print(f"   Manual Brier score calculation: {manual_brier:.6f}")

            # Compare with agent's result
            if test_metrics and "brier_score" in test_metrics:
                agent_brier = test_metrics["brier_score"]
                difference = abs(manual_brier - agent_brier)
                print(f"   Agent Brier score: {agent_brier:.6f}")
                print(f"   Difference: {difference:.10f}")

                if difference < 1e-10:
                    print("   ✅ Brier scores match perfectly!")
                else:
                    print("   ⚠️  Brier scores differ")
            else:
                print("   ❌ Could not compare - no agent Brier score")

    except Exception as e:
        print(f"   ❌ Manual verification failed: {e}")

    # 10. Summary and recommendations
    print("\n10. Summary and Recommendations:")

    has_test_brier = test_metrics and "brier_score" in test_metrics
    has_calib_brier = calib_metrics and "brier_score" in calib_metrics

    if has_test_brier and has_calib_brier:
        print("   ✅ SUCCESS: Brier score is being calculated and returned correctly!")
        print("   📊 Test Brier Score:", test_metrics["brier_score"])
        print("   📊 Calibration Brier Score:", calib_metrics["brier_score"])

        # Interpret the scores
        test_brier = test_metrics["brier_score"]
        calib_brier = calib_metrics["brier_score"]

        print("   📈 Interpretation:")
        for name, score in [("Test", test_brier), ("Calibration", calib_brier)]:
            if score < 0.05:
                print(f"     {name}: Excellent calibration")
            elif score < 0.10:
                print(f"     {name}: Good calibration")
            elif score < 0.15:
                print(f"     {name}: Fair calibration")
            else:
                print(f"     {name}: Poor calibration")
    else:
        print("   ❌ ISSUE: Brier score is not being calculated or returned properly")
        print("   🔧 Troubleshooting steps:")
        print("     1. Check the generated code in logs/h2o_automl_enhanced.py")
        print("     2. Verify the code includes Brier score calculation")
        print(
            "     3. Check that test_metrics and calibration_metrics are being returned"
        )
        print("     4. Ensure the agent is using the correct function name")

    print("\n✅ Brier score verification completed!")


def show_usage_examples():
    """Show examples of how to use the H2O agent to get Brier scores."""

    print("\n=== Usage Examples ===\n")

    print("1. Basic usage to get Brier scores:")
    print(
        """
    # Initialize agent
    ml_agent = H2OMLAgentEnhanced(
        model=llm,
        log=True,
        log_path="logs/",
        model_directory="models/"
    )
    
    # Run agent
    ml_agent.invoke_agent(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        X_calib=X_calib, y_calib=y_calib,
        user_instructions="Calculate Brier score for model calibration assessment."
    )
    
    # Get Brier scores
    test_metrics = ml_agent.get_test_metrics()
    calib_metrics = ml_agent.get_calibration_metrics()
    
    if test_metrics and 'brier_score' in test_metrics:
        print(f"Test Brier Score: {test_metrics['brier_score']:.6f}")
    
    if calib_metrics and 'brier_score' in calib_metrics:
        print(f"Calibration Brier Score: {calib_metrics['brier_score']:.6f}")
    """
    )

    print("\n2. Advanced usage with explicit Brier score requirements:")
    print(
        """
    user_instructions = '''
    Create an H2O AutoML model with the following requirements:
    
    1. MUST calculate Brier score for both test and calibration sets
    2. Brier score = mean((predicted_probability - actual_outcome)²)
    3. Use p1 (positive class probability) for binary classification
    4. Convert categorical targets to numeric (0/1) before calculation
    5. Store results in test_metrics['brier_score'] and calib_metrics['brier_score']
    
    The Brier score is essential for this analysis!
    '''
    """
    )

    print("\n3. Verification checklist:")
    print(
        """
    ✅ Check if test_metrics contains 'brier_score'
    ✅ Check if calib_metrics contains 'brier_score'
    ✅ Verify generated code includes Brier score calculation
    ✅ Compare with manual calculation for verification
    ✅ Interpret Brier scores (lower is better)
    """
    )


if __name__ == "__main__":
    ensure_brier_score_calculation()
    show_usage_examples()
