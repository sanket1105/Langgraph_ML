#!/usr/bin/env python3

import os

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

from h20 import H2OMLAgentEnhanced


def test_invoke_agent_fix():
    """Test that the invoke_agent method works correctly after the fix."""

    print("=== Testing invoke_agent Method Fix ===\n")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set!")
        print("Please set your OpenAI API key and try again.")
        return False

    # 1. Create sample data
    print("1. Creating sample data...")
    X, y = make_classification(
        n_samples=100,  # Small dataset for quick testing
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    # 2. Split the data
    X = df.drop(columns=["target"])
    y = df["target"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
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
    print("2. Initializing H2O ML Agent...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    ml_agent = H2OMLAgentEnhanced(
        model=llm,
        log=True,
        log_path=LOG_PATH,
        model_directory=MODEL_PATH,
        n_samples=10,  # Small sample for quick testing
        file_name="h2o_automl_enhanced.py",
        function_name="h2o_automl_enhanced",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=True,  # Skip recommendation step for faster testing
        bypass_explain_code=True,  # Skip explanation step for faster testing
        enable_mlflow=False,
        enable_optuna=False,  # Disable Optuna for faster testing
    )

    # 5. Test the invoke_agent method
    print("3. Testing invoke_agent method...")
    try:
        result = ml_agent.invoke_agent(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_calib=X_calib,
            y_calib=y_calib,
            user_instructions="Create a simple H2O AutoML model for binary classification.",
            max_retries=1,
        )

        print("✅ invoke_agent method executed successfully!")
        print(f"   Result type: {type(result)}")

        if hasattr(ml_agent, "response") and ml_agent.response:
            print("✅ Agent has response")
            print(f"   Response keys: {list(ml_agent.response.keys())}")
            return True
        else:
            print("❌ No response from agent")
            return False

    except Exception as e:
        print(f"❌ Error in invoke_agent method: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_invoke_agent_fix()
    if success:
        print("\n🎉 Test passed! The invoke_agent method fix is working correctly.")
    else:
        print("\n❌ Test failed! There are still issues with the invoke_agent method.")
