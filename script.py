import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import your H2O ML Agent (assuming it's saved as h2o_ml_agent_enhanced.py)
from h20 import H2OMLAgentEnhanced

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: OPENAI_API_KEY environment variable is not set!")
    print("Please set your OpenAI API key by running one of these commands:")
    print("  export OPENAI_API_KEY='your-api-key-here'")
    print("  or")
    print("  OPENAI_API_KEY='your-api-key-here' python script.py")
    print("\nYou can get an API key from: https://platform.openai.com/api-keys")
    exit(1)

# 1. Set up the language model
llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0.1  # or "gpt-4" for better performance
)

# 2. Create or Load your dataset
# Option A: Create sample data
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

# Option B: Load your own data
# df = pd.read_csv("your_data.csv")
# X = df.drop(columns=["target"])  # Replace "target" with your target column
# y = df["target"]

# 3. Split the data into train/test/calibration sets
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

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Calibration set size: {len(X_calib)}")

# 4. Set up directories
LOG_PATH = "logs/"
MODEL_PATH = "models/"
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 5. Initialize the H2O ML Agent
ml_agent = H2OMLAgentEnhanced(
    model=llm,
    log=True,
    log_path=LOG_PATH,
    model_directory=MODEL_PATH,
    n_samples=30,
    file_name="h2o_automl_enhanced.py",
    function_name="h2o_automl_enhanced",
    overwrite=True,
    human_in_the_loop=False,  # Set to True if you want to review steps
    bypass_recommended_steps=False,  # Set to True to skip recommendation step
    bypass_explain_code=False,  # Set to True to skip code explanation
    enable_mlflow=False,  # Set to True to enable MLflow logging
    mlflow_tracking_uri=None,
    mlflow_experiment_name="H2O AutoML Enhanced Experiment",
    mlflow_run_name="test_run_1",
    enable_optuna=True,  # Enable Optuna optimization
    optuna_n_trials=20,  # Number of optimization trials
    optuna_timeout=300,  # Timeout in seconds
)

# 6. Run the agent
print("Starting H2O ML Agent Enhanced...")

try:
    ml_agent.invoke_agent(
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
        max_retries=3,
    )

    print("✅ Agent execution completed successfully!")

    # Debug: Print the response structure
    print(
        f"\n🔍 Debug: Response keys: {list(ml_agent.response.keys()) if ml_agent.response else 'No response'}"
    )
    if ml_agent.response:
        for key, value in ml_agent.response.items():
            if key != "messages":  # Skip messages to avoid clutter
                print(f"  {key}: {type(value).__name__} - {str(value)[:100]}...")

    # 7. Get results
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    # Get leaderboard
    leaderboard = ml_agent.get_leaderboard()
    if leaderboard is not None:
        print("\n📊 H2O AutoML Leaderboard:")
        print(leaderboard.head())

    # Get best model ID
    best_model = ml_agent.get_best_model_id()
    if best_model:
        print(f"\n🏆 Best Model ID: {best_model}")

    # Get model path
    model_path = ml_agent.get_model_path()
    if model_path:
        print(f"\n💾 Model saved at: {model_path}")

    # Get optimization results
    optuna_results = ml_agent.get_optimization_results()
    if optuna_results:
        print(f"\n🔧 Optuna Optimization Results:")
        print(f"Best parameters: {optuna_results.get('best_params', 'N/A')}")
        print(f"Best value: {optuna_results.get('best_value', 'N/A')}")
        print(f"Number of trials: {optuna_results.get('n_trials', 'N/A')}")

    # Get test metrics
    test_metrics = ml_agent.get_test_metrics()
    if test_metrics:
        print(f"\n📈 Test Set Performance:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Get calibration metrics
    calib_metrics = ml_agent.get_calibration_metrics()
    if calib_metrics:
        print(f"\n🎯 Calibration Set Performance:")
        for metric, value in calib_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Get generated code
    print("\n💻 Generated H2O Function:")
    generated_code = ml_agent.get_h2o_train_function()
    if generated_code:
        print("Code saved to:", ml_agent.response.get("h2o_train_function_path", "N/A"))
        # Optionally display the code
        # print(generated_code[:500] + "..." if len(generated_code) > 500 else generated_code)

except Exception as e:
    print(f"❌ Error during execution: {str(e)}")
    print("Check the logs for more details.")

print("\n🏁 Process completed!")
