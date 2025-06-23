#!/usr/bin/env python3

import os

import pandas as pd

# Load environment variables
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from h20 import H2OMLAgentEnhanced

load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable is not set!")
    exit(1)

# Set up the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Create sample data
X, y = make_classification(
    n_samples=100,  # Smaller dataset for testing
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

# Split the data
X = df.drop(columns=["target"])
y = df["target"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Calibration set size: {len(X_calib)}")

# Set up directories
LOG_PATH = "logs/"
MODEL_PATH = "models/"
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

print("Initializing H2O ML Agent...")
try:
    # Initialize the H2O ML Agent with minimal settings
    ml_agent = H2OMLAgentEnhanced(
        model=llm,
        log=True,
        log_path=LOG_PATH,
        model_directory=MODEL_PATH,
        n_samples=10,  # Smaller sample for testing
        bypass_recommended_steps=True,  # Skip recommendation step
        bypass_explain_code=True,  # Skip explanation step
        enable_optuna=False,  # Disable Optuna for testing
        human_in_the_loop=False,
    )
    print("✅ Agent initialized successfully")
except Exception as e:
    print(f"❌ Error initializing agent: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("Running agent...")
try:
    # Run the agent with minimal settings
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
    print("✅ Agent execution completed")

    # Check results
    if hasattr(ml_agent, "response") and ml_agent.response:
        print("✅ Agent has response")
        print(f"Response keys: {list(ml_agent.response.keys())}")

        # Check for leaderboard
        leaderboard = ml_agent.get_leaderboard()
        if leaderboard is not None:
            print(f"✅ Leaderboard shape: {leaderboard.shape}")
            print(leaderboard.head())
        else:
            print("❌ No leaderboard found")
    else:
        print("❌ No response from agent")

except Exception as e:
    print(f"❌ Error running agent: {e}")
    import traceback

    traceback.print_exc()
