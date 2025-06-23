# Enhanced H2O Machine Learning Agent with Train/Test/Calib Split and Optuna Optimization

import json
import operator
import os
import warnings
from typing import Annotated, Any, Dict, Literal, Optional, Sequence, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer, Command
from matplotlib.backends.backend_pdf import PdfPages

from ai_logging import log_ai_function
from dataframe import get_dataframe_summary
from graph import (
    BaseAgent,
    create_coding_agent_graph,
    node_func_execute_agent_code_on_data,
    node_func_fix_agent_code,
    node_func_human_review,
    node_func_report_agent_outputs,
)
from h2o_doc import H2O_AUTOML_DOCUMENTATION
from parsers import PythonOutputParser
from regex import (
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
    get_generic_summary,
    relocate_imports_inside_function,
)

warnings.filterwarnings("ignore")

# Configure OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "").strip()
llm = ChatOpenAI(temperature=0)

# Define the agent name constant
AGENT_NAME = "h2o_ml_agent_enhanced"


def fix_code_indentation(code: str) -> str:
    """
    Fix indentation issues in generated Python code.

    Parameters
    ----------
    code : str
        The generated Python code that may have indentation issues.

    Returns
    -------
    str
        The code with proper indentation.
    """
    lines = code.split("\n")
    fixed_lines = []
    indent_level = 0
    in_function = False
    seen_imports = set()

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            fixed_lines.append("")
            continue

        # Check for function definition
        if stripped.startswith("def ") and ":" in stripped:
            in_function = True
            indent_level = 0
            fixed_lines.append(line)  # Function definition at base level
            continue

        # Check for class definition
        if stripped.startswith("class ") and ":" in stripped:
            in_function = False
            indent_level = 0
            fixed_lines.append(line)  # Class definition at base level
            continue

        # Handle import statements (should be at base level and avoid duplicates)
        if stripped.startswith(("import ", "from ")):
            if not in_function and stripped not in seen_imports:
                fixed_lines.append(line)  # Imports at base level
                seen_imports.add(stripped)
            continue

        # Check for conditional statements
        if stripped.startswith(
            (
                "if ",
                "elif ",
                "else:",
                "for ",
                "while ",
                "try:",
                "except ",
                "finally:",
                "with ",
            )
        ):
            if ":" in stripped:
                # This is a control structure
                if in_function:
                    fixed_lines.append("    " * (indent_level + 1) + stripped)
                    indent_level += 1
                else:
                    fixed_lines.append(stripped)
            else:
                # This is a continuation line
                if in_function:
                    fixed_lines.append("    " * (indent_level + 1) + stripped)
                else:
                    fixed_lines.append(stripped)
            continue

        # Check for return statement
        if stripped.startswith("return "):
            if in_function:
                fixed_lines.append("    " * (indent_level + 1) + stripped)
            else:
                fixed_lines.append(stripped)
            continue

        # Check for assignment or other statements
        if in_function:
            # Check if this line should reduce indentation
            if stripped in ["else:", "elif ", "except ", "finally:"]:
                indent_level = max(0, indent_level - 1)
            fixed_lines.append("    " * (indent_level + 1) + stripped)
        else:
            fixed_lines.append(stripped)

    # Clean up any empty conditional blocks
    cleaned_lines = []
    i = 0
    while i < len(fixed_lines):
        line = fixed_lines[i]
        stripped = line.strip()

        # Check for empty if blocks
        if stripped.startswith("if ") and ":" in stripped:
            # Look ahead to see if the next non-empty line is at the same or lower indentation
            next_indent = None
            j = i + 1
            while j < len(fixed_lines):
                next_line = fixed_lines[j].strip()
                if next_line:  # Found next non-empty line
                    next_indent = len(fixed_lines[j]) - len(fixed_lines[j].lstrip())
                    break
                j += 1

            current_indent = len(line) - len(line.lstrip())

            # If next line is at same or lower indentation, this is an empty block
            if next_indent is not None and next_indent <= current_indent:
                # Skip this empty if block
                i = j
                continue

        cleaned_lines.append(line)
        i += 1

    return "\n".join(cleaned_lines)


def validate_and_fix_code(code: str) -> str:
    """
    Validate and fix generated Python code to ensure it's syntactically correct.

    Parameters
    ----------
    code : str
        The generated Python code.

    Returns
    -------
    str
        The validated and fixed code.
    """
    import ast

    # First, try to parse the code to check for syntax errors
    try:
        ast.parse(code)
        return code  # Code is valid, return as is
    except SyntaxError as e:
        print(f"Syntax error detected: {e}")

        # Try to fix common issues
        lines = code.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                fixed_lines.append("")
                continue

            # Fix function definition missing colon
            if stripped.startswith("def ") and not stripped.endswith(":"):
                fixed_lines.append(line + ":")
                continue

            # Fix control structures missing colon
            if any(
                stripped.startswith(keyword)
                for keyword in [
                    "if ",
                    "elif ",
                    "else",
                    "for ",
                    "while ",
                    "try:",
                    "except ",
                    "finally:",
                    "with ",
                    "class ",
                ]
            ) and not stripped.endswith(":"):
                if not stripped.endswith(":"):
                    fixed_lines.append(line + ":")
                    continue

            # Fix indentation issues
            if stripped and not stripped.startswith(
                ("import ", "from ", "def ", "class ")
            ):
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces == 0:
                    # This should be indented inside the function
                    fixed_lines.append("    " + line)
                    continue

            fixed_lines.append(line)

        fixed_code = "\n".join(fixed_lines)

        # Try to parse the fixed code
        try:
            ast.parse(fixed_code)
            return fixed_code
        except SyntaxError:
            # If still invalid, return a minimal working version
            print("Could not fix syntax errors, using fallback code")
            return generate_fallback_code()

    return code


def generate_fallback_code() -> str:
    """
    Generate a minimal working H2O AutoML function as fallback.

    Returns
    -------
    str
        A minimal working H2O AutoML function.
    """
    return """
def h2o_automl_enhanced(train_data, test_data, calib_data, target_variable, feature_columns, enable_optuna=True, optuna_n_trials=50, optuna_timeout=300, model_directory=None, log_path=None, enable_mlflow=False, mlflow_tracking_uri=None, mlflow_experiment_name="H2O AutoML Enhanced", mlflow_run_name=None, **kwargs):
    import h2o
    from h2o.automl import H2OAutoML
    import pandas as pd
    import numpy as np
    import mapie
    from contextlib import nullcontext
    
    # Optional imports
    if enable_optuna:
        import optuna
        from optuna.samplers import TPESampler
    
    if enable_mlflow:
        import mlflow
        import mlflow.h2o
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        run_context = mlflow.start_run(run_name=mlflow_run_name)
    else:
        run_context = nullcontext()

    # Convert data to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    calib_df = pd.DataFrame(calib_data)

    with run_context as run:
        # Initialize H2O
        h2o.init()

        # Create H2OFrames
        train_h2o = h2o.H2OFrame(train_df)
        test_h2o = h2o.H2OFrame(test_df)
        calib_h2o = h2o.H2OFrame(calib_df)

        # Convert target variable to categorical if it's binary
        # Check if target has only 2 unique values by converting to pandas first
        target_values = train_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
        if len(set(target_values)) == 2:
            train_h2o[target_variable] = train_h2o[target_variable].asfactor()
            test_h2o[target_variable] = test_h2o[target_variable].asfactor()
            calib_h2o[target_variable] = calib_h2o[target_variable].asfactor()

        # Train AutoML model
        aml = H2OAutoML(
            max_runtime_secs=300,
            max_models=20,
            nfolds=5,
            seed=42,
            sort_metric="AUTO"
        )
        
        aml.train(x=feature_columns, y=target_variable, training_frame=train_h2o)
        
        # Evaluate on test set
        test_perf = aml.leader.model_performance(test_h2o)
        test_metrics = {}
        
        # Handle classification metrics
        try:
            if hasattr(test_perf, 'auc'):
                auc_value = test_perf.auc()
                test_metrics['auc'] = auc_value[0][0] if hasattr(auc_value, '__getitem__') else auc_value
        except:
            pass
            
        try:
            if hasattr(test_perf, 'logloss'):
                logloss_value = test_perf.logloss()
                test_metrics['logloss'] = logloss_value[0][0] if hasattr(logloss_value, '__getitem__') else logloss_value
        except:
            pass
            
        # Calculate Brier Score for binary classification
        try:
            if len(set(target_values)) == 2:  # Binary classification
                # Get predicted probabilities
                test_pred = aml.leader.predict(test_h2o)
                test_probs = test_pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()  # Probability of positive class
                test_actual = test_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                
                # Convert to numeric if categorical
                if test_actual.dtype == 'object':
                    test_actual = (test_actual == test_actual[0]).astype(int)
                
                # Calculate Brier Score
                brier_score = np.mean((test_probs - test_actual) ** 2)
                test_metrics['brier_score'] = brier_score
        except Exception as e:
            print(f"Could not calculate Brier score: {e}")
            
        # Handle regression metrics
        try:
            if hasattr(test_perf, 'rmse'):
                rmse_value = test_perf.rmse()
                test_metrics['rmse'] = rmse_value[0][0] if hasattr(rmse_value, '__getitem__') else rmse_value
        except:
            pass
            
        try:
            if hasattr(test_perf, 'mae'):
                mae_value = test_perf.mae()
                test_metrics['mae'] = mae_value[0][0] if hasattr(mae_value, '__getitem__') else mae_value
        except:
            pass
        
        # Evaluate on calibration set
        calib_perf = aml.leader.model_performance(calib_h2o)
        calib_metrics = {}
        
        # Handle classification metrics
        try:
            if hasattr(calib_perf, 'auc'):
                auc_value = calib_perf.auc()
                calib_metrics['auc'] = auc_value[0][0] if hasattr(auc_value, '__getitem__') else auc_value
        except:
            pass
            
        try:
            if hasattr(calib_perf, 'logloss'):
                logloss_value = calib_perf.logloss()
                calib_metrics['logloss'] = logloss_value[0][0] if hasattr(logloss_value, '__getitem__') else logloss_value
        except:
            pass
            
        # Calculate Brier Score for calibration set
        try:
            if len(set(target_values)) == 2:  # Binary classification
                # Get predicted probabilities
                calib_pred = aml.leader.predict(calib_h2o)
                calib_probs = calib_pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()  # Probability of positive class
                calib_actual = calib_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                
                # Convert to numeric if categorical
                if calib_actual.dtype == 'object':
                    calib_actual = (calib_actual == calib_actual[0]).astype(int)
                
                # Calculate Brier Score
                brier_score = np.mean((calib_probs - calib_actual) ** 2)
                calib_metrics['brier_score'] = brier_score
        except Exception as e:
            print(f"Could not calculate Brier score for calibration set: {e}")
            
        # Handle regression metrics
        try:
            if hasattr(calib_perf, 'rmse'):
                rmse_value = calib_perf.rmse()
                calib_metrics['rmse'] = rmse_value[0][0] if hasattr(rmse_value, '__getitem__') else rmse_value
        except:
            pass
            
        try:
            if hasattr(calib_perf, 'mae'):
                mae_value = calib_perf.mae()
                calib_metrics['mae'] = mae_value[0][0] if hasattr(mae_value, '__getitem__') else mae_value
        except:
            pass

        # Save model if directory provided
        model_path = None
        if model_directory or log_path:
            save_path = model_directory if model_directory else log_path
            model_path = h2o.save_model(model=aml.leader, path=save_path, force=True)

        # Get leaderboard
        leaderboard_df = aml.leaderboard.as_data_frame(use_multi_thread=True)
        # Compute Brier Score for all models in the leaderboard (binary classification only)
        if len(set(target_values)) == 2:
            brier_scores = []
            for model_id in leaderboard_df['model_id']:
                model = h2o.get_model(model_id)
                pred = model.predict(test_h2o)
                if 'p1' in pred.columns:
                    probs = pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()
                else:
                    # fallback to first probability column if p1 not present
                    prob_cols = [col for col in pred.columns if col.startswith('p')]
                    probs = pred[prob_cols[0]].as_data_frame(use_multi_thread=True).values.flatten()
                actual = test_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                if actual.dtype == 'object':
                    actual = (actual == actual[0]).astype(int)
                brier = np.mean((probs - actual) ** 2)
                brier_scores.append(brier)
            leaderboard_df['brier_score'] = brier_scores
            leaderboard_df = leaderboard_df.sort_values('brier_score', ascending=True).reset_index(drop=True)
        leaderboard_dict = leaderboard_df.to_dict()

        # Set best model as the one with the lowest Brier Score (if available)
        if 'brier_score' in leaderboard_df.columns:
            best_model_id = leaderboard_df.loc[0, 'model_id']
            best_model = h2o.get_model(best_model_id)
            if model_directory or log_path:
                save_path = model_directory if model_directory else log_path
                model_path = h2o.save_model(model=best_model, path=save_path, force=True)
            else:
                model_path = None
        else:
            best_model_id = aml.leader.model_id
            model_path = model_path if 'model_path' in locals() else None

        # Get model type and parameters
        model_type = type(best_model).__name__
        model_params = best_model.params if hasattr(best_model, "params") else {}

        # Prepare results
        results = {
            'leaderboard': leaderboard_dict,
            'best_model_id': best_model_id,
            'model_path': model_path,
            'test_metrics': test_metrics,
            'calibration_metrics': calib_metrics,
            'optimization_results': None,
            'model_results': {
                'model_flavor': 'H2O AutoML Enhanced',
                'model_path': model_path,
                'best_model_id': best_model_id,
                'test_performance': test_metrics,
                'calibration_performance': calib_metrics
            },
            'model_type': model_type,
            'model_parameters': model_params,
            'custom_models': {},  # Always include this key in fallback
        }

        return results
"""


LOG_PATH = os.path.join(os.getcwd(), "logs/")


class H2OMLAgentEnhanced(BaseAgent):
    """
    Enhanced H2O Machine Learning agent that accepts separate train/test/calibration datasets
    and uses Optuna for hyperparameter optimization.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the ML code.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the Python file for saving the generated code. Defaults to "h2o_automl_enhanced.py".
    function_name : str, optional
        Name of the function that performs the AutoML training. Defaults to "h2o_automl_enhanced".
    model_directory : str or None, optional
        Directory to save the H2O Machine Learning model. Defaults to None.
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of the code. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the recommended steps prompt. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the code-explanation step. Defaults to False.
    enable_mlflow : bool, default False
        Whether to enable MLflow logging.
    mlflow_tracking_uri : str or None
        MLflow tracking URI.
    mlflow_experiment_name : str
        Name of the MLflow experiment.
    mlflow_run_name : str, default None
        Custom name for the MLflow run.
    enable_optuna : bool, default True
        Whether to enable Optuna hyperparameter optimization.
    optuna_n_trials : int, default 50
        Number of Optuna trials for optimization.
    optuna_timeout : int, default 300
        Timeout in seconds for Optuna optimization.
    checkpointer : langgraph.checkpoint.memory.MemorySaver, optional
        A checkpointer object for saving the agent's state.

    Methods
    -------
    invoke_agent(X_train, y_train, X_test, y_test, X_calib, y_calib, user_instructions, ...)
        Trains an H2O AutoML model with the provided train/test/calibration splits.
    get_leaderboard()
        Retrieves the H2O AutoML leaderboard.
    get_best_model_id()
        Retrieves the best model ID.
    get_model_path()
        Retrieves the saved model path.
    get_optimization_results()
        Retrieves Optuna optimization results.
    get_test_metrics()
        Retrieves test set performance metrics.
    get_calibration_metrics()
        Retrieves calibration set performance metrics.
    get_model_type()
        Retrieves the model type.
    get_model_parameters()
        Retrieves the model parameters.
    empirical_cdf(self, data)
        Compute the empirical CDF for a 1D array-like of data.
    empirical_edf(self, data)
        Alias for empirical CDF (EDF and CDF are the same for empirical data).
    plot_cdf_vs_edf_for_leaderboard(self, leaderboard_df, test_h2o, target_variable, logs_dir="logs/")
        Plot CDF vs EDF for each model in the leaderboard and save all plots to a single PDF.
    plot_combined_reliability_curves(self, leaderboard_df, test_h2o, target_variable, custom_models=None, logs_dir="logs/")
        Plot reliability curves for H2O models first, then append custom models to the same figure.
        This ensures all models are included regardless of lookup issues.
    plot_custom_models_reliability_curves(self, custom_models, test_h2o, target_variable, logs_dir="logs/")
        Plot reliability curves for custom models only (no H2O models).
        This creates a focused view of custom model calibration.
    generate_reliability_curves_from_response(self, X_test, y_test, target_variable="target", logs_dir="logs/")
        Generate reliability curves PDFs for custom models and for both H2O and custom models,
        using the actual trained models from the agent's response.

    Examples
    --------
    ```python
    from langchain_openai import ChatOpenAI
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from ai_data_science_team.ml_agents import H2OMLAgentEnhanced

    llm = ChatOpenAI(model="gpt-4o-mini")

    # Load and split your data
    df = pd.read_csv("data/churn_data.csv")
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    ml_agent = H2OMLAgentEnhanced(
        model=llm,
        log=True,
        log_path=LOG_PATH,
        model_directory=MODEL_PATH,
        enable_optuna=True,
        optuna_n_trials=30
    )

    ml_agent.invoke_agent(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_calib=X_calib,
        y_calib=y_calib,
        user_instructions="Optimize for AUC with focus on recall. Use max 5 minutes for training.",
    )

    # Get results
    ml_agent.get_leaderboard()
    ml_agent.get_optimization_results()
    ml_agent.get_test_metrics()
    ml_agent.generate_reliability_curves_from_response(X_test, y_test)
    ```
    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="h2o_automl_enhanced.py",
        function_name="h2o_automl_enhanced",
        model_directory=None,
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        enable_mlflow=False,
        mlflow_tracking_uri=None,
        mlflow_experiment_name="H2O AutoML Enhanced",
        mlflow_run_name=None,
        enable_optuna=True,
        optuna_n_trials=50,
        optuna_timeout=300,
        checkpointer: Optional[Checkpointer] = None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "model_directory": model_directory,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "enable_mlflow": enable_mlflow,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
            "enable_optuna": enable_optuna,
            "optuna_n_trials": optuna_n_trials,
            "optuna_timeout": optuna_timeout,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Creates the compiled graph for the agent."""
        self.response = None
        return make_h2o_ml_agent_enhanced(**self._params)

    def update_params(self, **kwargs):
        """Updates the agent's parameters and rebuilds the compiled graph."""
        for k, v in kwargs.items():
            self._params[k] = v
        self._compiled_graph = self._make_compiled_graph()

    def invoke_agent(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_calib: pd.DataFrame,
        y_calib: pd.Series,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Trains an H2O AutoML model with separate train/test/calibration datasets.
        """
        # Combine features and target for each split
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        calib_data = pd.concat([X_calib, y_calib], axis=1)

        # Prepare the complete state that the graph expects
        initial_state = {
            "messages": [],  # Will be populated during execution
            "user_instructions": user_instructions or "",
            "recommended_steps": "",  # Will be populated by recommend_ml_steps
            "train_data": train_data.to_dict(),
            "test_data": test_data.to_dict(),
            "calib_data": calib_data.to_dict(),
            "target_variable": y_train.name,
            "feature_columns": X_train.columns.tolist(),
            "leaderboard": {},  # Will be populated by execute_h2o_code
            "best_model_id": "",  # Will be populated by execute_h2o_code
            "model_path": "",  # Will be populated by execute_h2o_code
            "model_results": {},  # Will be populated by execute_h2o_code
            "optimization_results": {},  # Will be populated by execute_h2o_code
            "test_metrics": {},  # Will be populated by execute_h2o_code
            "calibration_metrics": {},  # Will be populated by execute_h2o_code
            "all_datasets_summary": "",  # Will be populated by recommend_ml_steps
            "h2o_train_function": "",  # Will be populated by create_h2o_code
            "h2o_train_function_path": "",  # Will be populated by create_h2o_code
            "h2o_train_file_name": "",  # Will be populated by create_h2o_code
            "h2o_train_function_name": "",  # Will be populated by create_h2o_code
            "h2o_train_error": "",  # Will be populated if there are errors
            "max_retries": max_retries,
            "retry_count": retry_count,
        }

        result = self._compiled_graph.invoke(
            input=initial_state,
            config=None,
        )
        self.response = result

        # Automatically generate reliability curves PDFs after training
        try:
            self.generate_reliability_curves_from_response(
                X_test=X_test,
                y_test=y_test,
                target_variable=y_train.name if hasattr(y_train, "name") else "target",
                logs_dir=self._params.get("log_path", "logs/"),
            )
        except Exception as e:
            print(f"[Warning] Could not generate reliability curves automatically: {e}")

        return result

    def get_leaderboard(self):
        """Returns the H2O AutoML leaderboard as a DataFrame."""
        if self.response and "leaderboard" in self.response:
            return pd.DataFrame(self.response["leaderboard"])
        return None

    def get_best_model_id(self):
        """Returns the best model ID from the unified leaderboard (H2O or custom models)."""
        if self.response and "leaderboard" in self.response:
            import pandas as pd

            # Get the leaderboard
            leaderboard_data = self.response["leaderboard"]
            if isinstance(leaderboard_data, dict):
                leaderboard_df = pd.DataFrame(leaderboard_data)
            else:
                leaderboard_df = leaderboard_data

            # Return the first model_id (best model) from the sorted leaderboard
            if not leaderboard_df.empty and "model_id" in leaderboard_df.columns:
                return leaderboard_df.iloc[0]["model_id"]

        # Fallback to the original best_model_id if available
        if self.response and "best_model_id" in self.response:
            return self.response["best_model_id"]

        return None

    def get_model_path(self):
        """Returns the file path to the saved best model."""
        if self.response and "model_path" in self.response:
            return self.response["model_path"]
        return None

    def get_optimization_results(self):
        """Returns Optuna optimization results."""
        if self.response and "optimization_results" in self.response:
            return self.response["optimization_results"]
        return None

    def get_test_metrics(self):
        """Returns test set performance metrics."""
        if self.response and "test_metrics" in self.response:
            return self.response["test_metrics"]
        return None

    def get_calibration_metrics(self):
        """Returns calibration set performance metrics."""
        if self.response and "calibration_metrics" in self.response:
            return self.response["calibration_metrics"]
        return None

    def get_h2o_train_function(self, markdown=False):
        """Retrieves the H2O AutoML function code generated by the agent."""
        if self.response and "h2o_train_function" in self.response:
            code = self.response["h2o_train_function"]
            if markdown:
                return Markdown(f"```python\n{code}\n```")
            return code
        return None

    def get_model_type(self):
        if self.response and "model_type" in self.response:
            return self.response["model_type"]
        return None

    def get_model_parameters(self):
        if self.response and "model_parameters" in self.response:
            return self.response["model_parameters"]
        return None

    def empirical_cdf(self, data):
        """Compute the empirical CDF for a 1D array-like of data."""
        import numpy as np

        data = np.sort(np.asarray(data))
        n = data.size
        y = np.arange(1, n + 1) / n
        return data, y

    def empirical_edf(self, data):
        """Alias for empirical CDF (EDF and CDF are the same for empirical data)."""
        return self.empirical_cdf(data)

    def plot_cdf_vs_edf_for_leaderboard(
        self, leaderboard_df, test_h2o, target_variable, logs_dir="logs/"
    ):
        """
        Plot CDF vs EDF for each model in the leaderboard and save all plots to a single PDF.
        Returns the path to the PDF file.
        """
        import h2o

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        pdf_path = os.path.join(logs_dir, "cdf_vs_edf_leaderboard.pdf")
        with PdfPages(pdf_path) as pdf:
            for idx, model_id in enumerate(leaderboard_df["model_id"]):
                try:
                    model = h2o.get_model(model_id)
                    pred = model.predict(test_h2o)
                    if "p1" in pred.columns:
                        probs = (
                            pred["p1"]
                            .as_data_frame(use_multi_thread=True)
                            .values.flatten()
                        )
                    else:
                        prob_cols = [col for col in pred.columns if col.startswith("p")]
                        probs = (
                            pred[prob_cols[0]]
                            .as_data_frame(use_multi_thread=True)
                            .values.flatten()
                        )
                    actual = (
                        test_h2o[target_variable]
                        .as_data_frame(use_multi_thread=True)
                        .values.flatten()
                    )
                    # Convert to numeric if categorical
                    if hasattr(actual, "dtype") and actual.dtype == "object":
                        actual = (actual == actual[0]).astype(int)
                    else:
                        actual = actual.astype(int)
                    # Empirical Distribution Function (EDF)
                    sorted_probs = np.sort(probs)
                    edf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
                    # Cumulative Distribution Function (CDF) of actuals (for positive class)
                    sorted_actual = np.sort(actual)
                    cdf = np.arange(1, len(sorted_actual) + 1) / len(sorted_actual)
                    # Plot
                    plt.figure(figsize=(7, 5))
                    plt.plot(
                        sorted_probs,
                        edf,
                        label="EDF (Predicted Probabilities)",
                        color="blue",
                    )
                    plt.plot(
                        sorted_actual,
                        cdf,
                        label="CDF (Actual Outcomes)",
                        color="orange",
                    )
                    plt.title(f"CDF vs EDF for Model: {model_id}")
                    plt.xlabel("Probability / Outcome")
                    plt.ylabel("Cumulative Fraction")
                    plt.legend()
                    plt.grid(True)
                    pdf.savefig()
                    plt.close()
                except Exception as e:
                    print(f"Could not plot CDF vs EDF for model {model_id}: {e}")
        return pdf_path

    def plot_combined_reliability_curves(
        self,
        leaderboard_df,
        test_h2o,
        target_variable,
        custom_models=None,
        logs_dir="logs/",
    ):
        """
        Plot reliability curves for H2O models first, then append custom models to the same figure.
        This ensures all models are included regardless of lookup issues.
        """
        import os

        import h2o
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.calibration import calibration_curve

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        pdf_path = os.path.join(logs_dir, "reliability_curves_leaderboard.pdf")

        # Separate H2O models and custom models
        h2o_models = []
        custom_model_entries = []

        for idx, row in leaderboard_df.iterrows():
            model_id = row["model_id"]
            try:
                # Try to get H2O model
                model = h2o.get_model(model_id)
                h2o_models.append((idx, model_id, model))
            except Exception:
                # If not H2O model, it's a custom model
                custom_model_entries.append((idx, model_id))

        # Calculate total models for subplot layout
        total_models = len(h2o_models) + len(custom_model_entries)
        if total_models == 0:
            print("No models in leaderboard to plot reliability curves.")
            return None

        n_cols = min(3, total_models)
        n_rows = (total_models + n_cols - 1) // n_cols
        plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        plot_idx = 0

        # Plot H2O models first
        for idx, model_id, model in h2o_models:
            plt.subplot(n_rows, n_cols, plot_idx + 1)
            plot_idx += 1

            pred = model.predict(test_h2o)
            if "p1" in pred.columns:
                probs = pred["p1"].as_data_frame(use_multi_thread=True).values.flatten()
            else:
                prob_cols = [col for col in pred.columns if col.startswith("p")]
                probs = (
                    pred[prob_cols[0]]
                    .as_data_frame(use_multi_thread=True)
                    .values.flatten()
                )

            actual = (
                test_h2o[target_variable]
                .as_data_frame(use_multi_thread=True)
                .values.flatten()
            )
            if hasattr(actual, "dtype") and actual.dtype == "object":
                actual = (actual == actual[0]).astype(int)
            else:
                actual = actual.astype(int)

            prob_true, prob_pred = calibration_curve(actual, probs, n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label="Test")
            plt.plot([0, 1], [0, 1], "r--", label="Perfectly Calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"H2O: {model_id[:25]}{'...' if len(model_id) > 25 else ''}")
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot custom models
        if custom_models:
            for idx, model_id in custom_model_entries:
                plt.subplot(n_rows, n_cols, plot_idx + 1)
                plot_idx += 1

                # Find the custom model by trying different matching strategies
                model = None
                for key, value in custom_models.items():
                    if (
                        key.strip().lower() == model_id.strip().lower()
                        or key.strip().lower() in model_id.strip().lower()
                        or model_id.strip().lower() in key.strip().lower()
                    ):
                        model = value
                        break

                if model is not None:
                    try:
                        test_df = test_h2o.as_data_frame(use_multi_thread=True)
                        if target_variable in test_df.columns:
                            test_df = test_df.drop(columns=[target_variable])
                        probs = model.predict_proba(test_df)[:, 1]

                        actual = (
                            test_h2o[target_variable]
                            .as_data_frame(use_multi_thread=True)
                            .values.flatten()
                        )
                        if hasattr(actual, "dtype") and actual.dtype == "object":
                            actual = (actual == actual[0]).astype(int)
                        else:
                            actual = actual.astype(int)

                        prob_true, prob_pred = calibration_curve(
                            actual, probs, n_bins=10
                        )
                        plt.plot(prob_pred, prob_true, marker="o", label="Test")
                        plt.plot([0, 1], [0, 1], "r--", label="Perfectly Calibrated")
                        plt.xlabel("Mean Predicted Probability")
                        plt.ylabel("Fraction of Positives")
                        plt.title(
                            f"Custom: {model_id[:25]}{'...' if len(model_id) > 25 else ''}"
                        )
                        plt.ylim([0, 1])
                        plt.xlim([0, 1])
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                    except Exception as e:
                        plt.text(
                            0.5,
                            0.5,
                            f"Error plotting\n{model_id}\n{str(e)}",
                            ha="center",
                            va="center",
                            transform=plt.gca().transAxes,
                        )
                        plt.title(
                            f"Custom: {model_id[:25]}{'...' if len(model_id) > 25 else ''}"
                        )
                else:
                    plt.text(
                        0.5,
                        0.5,
                        f"Model not found\n{model_id}",
                        ha="center",
                        va="center",
                        transform=plt.gca().transAxes,
                    )
                    plt.title(
                        f"Custom: {model_id[:25]}{'...' if len(model_id) > 25 else ''}"
                    )

        plt.suptitle("Reliability Curves (Calibration) for All Models", fontsize=16)
        plt.tight_layout()
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()
        return pdf_path

    def plot_reliability_curves_for_leaderboard(
        self,
        leaderboard_df,
        test_h2o,
        target_variable,
        custom_models=None,
        logs_dir="logs/",
    ):
        """
        Plot reliability curves for all models in the leaderboard (H2O and custom models) and for custom models only.
        Saves two PDFs: one for the leaderboard (reliability_curves_leaderboard.pdf) and one for custom models (custom_models_reliability_curves.pdf).
        Returns a dict with both PDF paths.
        """
        import os

        import h2o
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.calibration import calibration_curve

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        pdf_path = os.path.join(logs_dir, "reliability_curves_leaderboard.pdf")

        n_models = len(leaderboard_df)
        if n_models == 0:
            print("No models in leaderboard to plot reliability curves.")
            return None
        n_cols = min(3, n_models)  # Max 3 columns
        n_rows = (n_models + n_cols - 1) // n_cols

        # Build a normalized custom model lookup for robust matching
        custom_model_lookup = {}
        if custom_models:
            for k, v in custom_models.items():
                norm_k = k.strip().lower()
                custom_model_lookup[norm_k] = v

        plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        for idx, model_id in enumerate(leaderboard_df["model_id"]):
            plt.subplot(n_rows, n_cols, idx + 1)
            try:
                # Try to get H2O model first
                model = h2o.get_model(model_id)
                pred = model.predict(test_h2o)
                if "p1" in pred.columns:
                    probs = (
                        pred["p1"].as_data_frame(use_multi_thread=True).values.flatten()
                    )
                else:
                    prob_cols = [col for col in pred.columns if col.startswith("p")]
                    probs = (
                        pred[prob_cols[0]]
                        .as_data_frame(use_multi_thread=True)
                        .values.flatten()
                    )
            except Exception:
                # If H2O model not found, try custom model with robust matching
                norm_id = str(model_id).strip().lower()
                if custom_model_lookup and norm_id in custom_model_lookup:
                    model = custom_model_lookup[norm_id]
                    test_df = test_h2o.as_data_frame(use_multi_thread=True)
                    if target_variable in test_df.columns:
                        test_df = test_df.drop(columns=[target_variable])
                    probs = model.predict_proba(test_df)[:, 1]
                else:
                    print(
                        f"Model {model_id} not found in H2O registry or custom models"
                    )
                    continue
            actual = (
                test_h2o[target_variable]
                .as_data_frame(use_multi_thread=True)
                .values.flatten()
            )
            if hasattr(actual, "dtype") and actual.dtype == "object":
                actual = (actual == actual[0]).astype(int)
            else:
                actual = actual.astype(int)
            prob_true, prob_pred = calibration_curve(actual, probs, n_bins=10)
            plt.plot(prob_pred, prob_true, marker="o", label="Test")
            plt.plot([0, 1], [0, 1], "r--", label="Perfectly Calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(f"{model_id[:30]}{'...' if len(model_id) > 30 else ''}")
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.legend()
            plt.grid(True, alpha=0.3)
        plt.suptitle("Reliability Curves (Calibration) for All Models", fontsize=16)
        plt.tight_layout()
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Always generate custom models reliability curves PDF as well
        custom_pdf_path = None
        if custom_models:
            custom_pdf_path = self.plot_custom_models_reliability_curves(
                custom_models=custom_models,
                test_h2o=test_h2o,
                target_variable=target_variable,
                logs_dir=logs_dir,
            )
        else:
            print(
                "No custom models provided for separate custom models reliability curves PDF."
            )

        return {"leaderboard_pdf": pdf_path, "custom_models_pdf": custom_pdf_path}

    def train_and_compare_all_models(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        h2o_leaderboard_df,
        test_h2o,
        target_variable,
        logs_dir="logs/",
    ):
        """
        Trains and Optuna-tunes XGBoost, CatBoost, LightGBM, RandomForest, Logistic Regression (L1 & L2),
        merges their results with the H2O leaderboard, and plots reliability curves for all models.
        Returns the unified leaderboard and PDF path.
        """
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        import optuna
        import pandas as pd
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from sklearn.calibration import calibration_curve
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from xgboost import XGBClassifier

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        custom_models = {}
        custom_results = []

        # Helper for Optuna tuning
        def tune_model(objective, n_trials=30, timeout=300):
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            return study.best_trial.params

        # Random Forest
        def rf_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            }
            clf = RandomForestClassifier(**params, random_state=42)
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, probas)

        rf_params = tune_model(rf_objective)
        rf = RandomForestClassifier(**rf_params, random_state=42)
        rf.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        custom_models["Random Forest"] = rf

        # XGBoost
        def xgb_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
            clf = XGBClassifier(**params, random_state=42)
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, probas)

        xgb_params = tune_model(xgb_objective)
        xgb = XGBClassifier(**xgb_params, random_state=42)
        xgb.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        custom_models["XGBoost"] = xgb

        # CatBoost
        def cat_objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "verbose": 0,
            }
            clf = CatBoostClassifier(**params, random_state=42)
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, probas)

        cat_params = tune_model(cat_objective)
        cat = CatBoostClassifier(**cat_params, random_state=42, verbose=0)
        cat.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        custom_models["CatBoost"] = cat

        # LightGBM
        def lgbm_objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }
            clf = LGBMClassifier(**params, random_state=42)
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, probas)

        lgbm_params = tune_model(lgbm_objective)
        lgbm = LGBMClassifier(**lgbm_params, random_state=42)
        lgbm.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        custom_models["LightGBM"] = lgbm

        # Logistic Regression L1
        def lr_l1_objective(trial):
            params = {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "penalty": "l1",
                "solver": "liblinear",
            }
            clf = LogisticRegression(**params, random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, probas)

        lr_l1_params = tune_model(lr_l1_objective)
        lr_l1 = LogisticRegression(**lr_l1_params, random_state=42, max_iter=1000)
        lr_l1.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        custom_models["Logistic Regression (L1)"] = lr_l1

        # Logistic Regression L2
        def lr_l2_objective(trial):
            params = {
                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                "penalty": "l2",
                "solver": "liblinear",
            }
            clf = LogisticRegression(**params, random_state=42, max_iter=1000)
            clf.fit(X_train, y_train)
            probas = clf.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, probas)

        lr_l2_params = tune_model(lr_l2_objective)
        lr_l2 = LogisticRegression(**lr_l2_params, random_state=42, max_iter=1000)
        lr_l2.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        custom_models["Logistic Regression (L2)"] = lr_l2

        # Collect results for custom models
        for name, model in custom_models.items():
            probas = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probas)
            brier = brier_score_loss(y_test, probas)
            logloss = log_loss(y_test, probas)
            custom_results.append(
                {
                    "model_id": name,
                    "auc": auc,
                    "brier_score": brier,
                    "logloss": logloss,
                    "probas": probas,
                }
            )

        # Prepare unified leaderboard
        # Collect all metric columns from both H2O and custom models
        h2o_cols = set(h2o_leaderboard_df.columns)
        custom_cols = set()
        for res in custom_results:
            custom_cols.update(res.keys())
        custom_cols.discard("probas")
        all_cols = list({"model_id"} | (h2o_cols | custom_cols) - {"probas"})
        if "model_parameters" not in all_cols:
            all_cols.append("model_parameters")

        # Build the leaderboard with all columns
        leaderboard = h2o_leaderboard_df.copy()
        leaderboard = leaderboard.reindex(columns=all_cols)
        for res in custom_results:
            row = {
                col: res.get(col, np.nan)
                for col in all_cols
                if col != "model_parameters"
            }
            # Add model parameters for custom models
            model_obj = custom_models.get(res["model_id"])
            if model_obj is not None:
                row["model_parameters"] = json.dumps(model_obj.get_params())
            else:
                row["model_parameters"] = np.nan
            leaderboard = pd.concat(
                [leaderboard, pd.DataFrame([row])], ignore_index=True
            )
        # Add model parameters for H2O models
        for idx, model_id in enumerate(h2o_leaderboard_df["model_id"]):
            try:
                import h2o

                model = h2o.get_model(model_id)
                leaderboard.at[idx, "model_parameters"] = json.dumps(model.params)
            except Exception:
                leaderboard.at[idx, "model_parameters"] = np.nan

        # Optionally, sort by brier_score or auc if present
        if "brier_score" in leaderboard.columns:
            leaderboard = leaderboard.sort_values(
                "brier_score", ascending=True
            ).reset_index(drop=True)
        elif "auc" in leaderboard.columns:
            leaderboard = leaderboard.sort_values("auc", ascending=False).reset_index(
                drop=True
            )

        # Plot reliability curves for all models (existing code)
        n_models = len(custom_results) + len(h2o_leaderboard_df)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        # H2O models
        for idx, model_id in enumerate(leaderboard["model_id"]):
            # ... (rest of your plotting code)
            pass

        # Return leaderboard and pdf_path as before
        return leaderboard, pdf_path

    def plot_custom_models_reliability_curves(
        self, custom_models, test_h2o, target_variable, logs_dir="logs/"
    ):
        """
        Plot reliability curves for custom models only (no H2O models).
        This creates a focused view of custom model calibration.
        """
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.calibration import calibration_curve

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        pdf_path = os.path.join(logs_dir, "custom_models_reliability_curves.pdf")

        if not custom_models:
            print("No custom models provided for plotting.")
            return None

        # Calculate subplot layout
        n_models = len(custom_models)
        n_cols = min(3, n_models)  # Max 3 columns
        n_rows = (n_models + n_cols - 1) // n_cols

        plt.figure(figsize=(5 * n_cols, 5 * n_rows))

        for idx, (model_name, model) in enumerate(custom_models.items()):
            plt.subplot(n_rows, n_cols, idx + 1)

            try:
                # Get test data (excluding target variable)
                test_df = test_h2o.as_data_frame(use_multi_thread=True)
                if target_variable in test_df.columns:
                    test_df = test_df.drop(columns=[target_variable])

                # Get predictions
                probs = model.predict_proba(test_df)[:, 1]

                # Get actual values
                actual = (
                    test_h2o[target_variable]
                    .as_data_frame(use_multi_thread=True)
                    .values.flatten()
                )
                if hasattr(actual, "dtype") and actual.dtype == "object":
                    actual = (actual == actual[0]).astype(int)
                else:
                    actual = actual.astype(int)

                # Calculate calibration curve
                prob_true, prob_pred = calibration_curve(actual, probs, n_bins=10)

                # Plot
                plt.plot(
                    prob_pred,
                    prob_true,
                    marker="o",
                    label="Test",
                    linewidth=2,
                    markersize=6,
                )
                plt.plot([0, 1], [0, 1], "r--", label="Perfectly Calibrated", alpha=0.7)

                # Calculate and display Brier score
                from sklearn.metrics import brier_score_loss

                brier_score = brier_score_loss(actual, probs)
                plt.text(
                    0.05,
                    0.95,
                    f"Brier Score: {brier_score:.4f}",
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

                plt.xlabel("Mean Predicted Probability")
                plt.ylabel("Fraction of Positives")
                plt.title(f"{model_name}")
                plt.ylim([0, 1])
                plt.xlim([0, 1])
                plt.legend()
                plt.grid(True, alpha=0.3)

            except Exception as e:
                plt.text(
                    0.5,
                    0.5,
                    f"Error plotting\n{model_name}\n{str(e)}",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    color="red",
                )
                plt.title(f"{model_name} (Error)")
                plt.xlim([0, 1])
                plt.ylim([0, 1])

        plt.suptitle(
            "Reliability Curves for Custom Models", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Custom models reliability curves saved to: {pdf_path}")
        return pdf_path

    def generate_reliability_curves_from_response(
        self, X_test, y_test, target_variable="target", logs_dir="logs/"
    ):
        """
        Generate reliability curves PDFs for custom models and for both H2O and custom models,
        using the actual trained models from the agent's response.
        Args:
            X_test: Test features (pd.DataFrame)
            y_test: Test target (pd.Series)
            target_variable: Name of the target column (default: "target")
            logs_dir: Directory to save the PDFs (default: "logs/")
        """
        import os

        import h2o

        os.makedirs(logs_dir, exist_ok=True)

        # Prepare test H2OFrame
        test_h2o = h2o.H2OFrame(X_test)
        test_h2o[target_variable] = h2o.H2OFrame(y_test.to_frame())

        # Get actual trained custom models
        custom_models = (
            self.response["custom_models"]
            if self.response and "custom_models" in self.response
            else None
        )
        if not custom_models:
            print("No custom models found in agent response. Run the agent first.")
            return

        # Generate reliability curves PDF for custom models
        custom_pdf_path = self.plot_custom_models_reliability_curves(
            custom_models=custom_models,
            test_h2o=test_h2o,
            target_variable=target_variable,
            logs_dir=logs_dir,
        )
        print(f"Custom models reliability curves saved to: {custom_pdf_path}")

        # Generate combined PDF (H2O + Custom models)
        leaderboard = self.get_leaderboard()
        if leaderboard is not None:
            combined_pdf_path = self.plot_combined_reliability_curves(
                leaderboard_df=leaderboard,
                test_h2o=test_h2o,
                target_variable=target_variable,
                custom_models=custom_models,
                logs_dir=logs_dir,
            )
            print(f"Combined reliability curves saved to: {combined_pdf_path}")
        else:
            print("No leaderboard found in agent response. Run the agent first.")


def make_h2o_ml_agent_enhanced(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="h2o_automl_enhanced.py",
    function_name="h2o_automl_enhanced",
    model_directory=None,
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    enable_mlflow=False,
    mlflow_tracking_uri=None,
    mlflow_experiment_name="H2O AutoML Enhanced",
    mlflow_run_name=None,
    enable_optuna=True,
    optuna_n_trials=50,
    optuna_timeout=300,
    checkpointer=None,
):
    """
    Creates an enhanced H2O ML agent with train/test/calib splits and Optuna optimization.
    """
    llm = model

    # Handle logging directory
    if log:
        if log_path is None:
            log_path = "logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Check required libraries
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError as e:
        raise ImportError(
            "The 'h2o' library is not installed. Please install it using: pip install h2o"
        ) from e

    if enable_optuna:
        try:
            import optuna
        except ImportError as e:
            raise ImportError(
                "The 'optuna' library is not installed. Please install it using: pip install optuna"
            ) from e

    if human_in_the_loop:
        if checkpointer is None:
            print(
                "Human in the loop is enabled. Setting checkpointer to MemorySaver()."
            )
            checkpointer = MemorySaver()

    # Define Enhanced GraphState
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        train_data: dict
        test_data: dict
        calib_data: dict
        target_variable: str
        feature_columns: list
        leaderboard: dict
        best_model_id: str
        model_path: str
        model_results: dict
        optimization_results: dict
        test_metrics: dict
        calibration_metrics: dict
        all_datasets_summary: str
        h2o_train_function: str
        h2o_train_function_path: str
        h2o_train_file_name: str
        h2o_train_function_name: str
        h2o_train_error: str
        max_retries: int
        retry_count: int

    # 1) Recommend ML steps
    def recommend_ml_steps(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND MACHINE LEARNING STEPS")

        recommend_steps_prompt = PromptTemplate(
            template="""
                You are an Enhanced H2O AutoML Expert with Optuna optimization capabilities.
                
                User instructions:
                    {user_instructions}

                Dataset Summary:
                    {all_datasets_summary}
                    
                H2O AutoML Documentation:
                    {h2o_automl_documentation}
                
                Optuna Integration: {enable_optuna}
                Optuna Trials: {optuna_n_trials}
                Optuna Timeout: {optuna_timeout} seconds

                Please recommend steps for H2O AutoML with the following considerations:
                
                1. **Data Strategy**: We have separate train/test/calibration splits - recommend how to best utilize each.
                2. **H2O AutoML Parameters**: Recommend parameters for maximum performance.
                3. **Optuna Optimization**: If enabled, suggest hyperparameters to optimize and their ranges.
                4. **Evaluation Strategy**: Recommend metrics and validation approaches, including Brier Score for probabilistic calibration.
                5. **Model Selection**: Suggest algorithms and ensembling strategies.
                
                Focus on:
                - Maximizing predictive performance on test set
                - Using calibration set for model calibration/threshold tuning
                - Efficient hyperparameter optimization with Optuna
                - Robust evaluation metrics including Brier Score for probability calibration
                - Model interpretability and reliability
                
                Return as a numbered list with specific recommendations.
            """,
            input_variables=[
                "user_instructions",
                "all_datasets_summary",
                "h2o_automl_documentation",
                "enable_optuna",
                "optuna_n_trials",
                "optuna_timeout",
            ],
        )

        # Create dataset summaries
        train_df = pd.DataFrame.from_dict(state.get("train_data"))
        test_df = pd.DataFrame.from_dict(state.get("test_data"))
        calib_df = pd.DataFrame.from_dict(state.get("calib_data"))

        # Create a dictionary of DataFrames with descriptive names
        dataframes_dict = {"Train": train_df, "Test": test_df, "Calibration": calib_df}

        all_datasets_summary = get_dataframe_summary(
            dataframes_dict,
            n_sample=n_samples,
        )
        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "all_datasets_summary": all_datasets_summary_str,
                "h2o_automl_documentation": H2O_AUTOML_DOCUMENTATION,
                "enable_optuna": enable_optuna,
                "optuna_n_trials": optuna_n_trials,
                "optuna_timeout": optuna_timeout,
            }
        )

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(),
                heading="# Enhanced ML Steps with Train/Test/Calib Split:",
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    # 2) Create enhanced H2O code with Optuna
    def create_h2o_code(state: GraphState):
        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            train_df = pd.DataFrame.from_dict(state.get("train_data"))
            test_df = pd.DataFrame.from_dict(state.get("test_data"))
            calib_df = pd.DataFrame.from_dict(state.get("calib_data"))

            # Create a dictionary of DataFrames with descriptive names
            dataframes_dict = {
                "Train": train_df,
                "Test": test_df,
                "Calibration": calib_df,
            }

            all_datasets_summary = get_dataframe_summary(
                dataframes_dict,
                n_sample=n_samples,
            )
            all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")

        print("    * CREATE ENHANCED H2O AUTOML CODE WITH OPTUNA")

        code_prompt = PromptTemplate(
            template="""
            You are an Enhanced H2O AutoML agent with Optuna optimization capabilities.
            Create a Python function named {function_name} that:
            
            1. Accepts separate train/test/calibration datasets
            2. Uses H2O AutoML for model training
            3. Optionally uses Optuna for hyperparameter optimization
            4. Evaluates on test and calibration sets
            5. Returns comprehensive results
            
            Parameters to consider:
            - enable_optuna: {enable_optuna}
            - optuna_n_trials: {optuna_n_trials}
            - optuna_timeout: {optuna_timeout}
            - model_directory: {model_directory}
            - log_path: {log_path}
            - enable_mlflow: {enable_mlflow}
            
            User Instructions:
                {user_instructions}
            
            Recommended Steps:
                {recommended_steps}

            Dataset Summary:
                {all_datasets_summary}
                
            Target Variable: {target_variable}
            Feature Columns: {feature_columns}

            Return ONLY the Python function code in ```python``` format. Use this EXACT structure:

            ```python
            def {function_name}(train_data, test_data, calib_data, target_variable, feature_columns, enable_optuna=True, optuna_n_trials=50, optuna_timeout=300, model_directory=None, log_path=None, enable_mlflow=False, mlflow_tracking_uri=None, mlflow_experiment_name="H2O AutoML Enhanced", mlflow_run_name=None, **kwargs):
                import h2o
                from h2o.automl import H2OAutoML
                import pandas as pd
                import numpy as np
                import json
                from typing import Dict, Any, List, Optional
                from contextlib import nullcontext
                
                # Optional imports for Optuna
                if enable_optuna:
                    import optuna
                    from optuna.samplers import TPESampler
                
                # Optional imports for MLflow
                if enable_mlflow:
                    import mlflow
                    import mlflow.h2o
                    if mlflow_tracking_uri:
                        mlflow.set_tracking_uri(mlflow_tracking_uri)
                    mlflow.set_experiment(mlflow_experiment_name)
                    run_context = mlflow.start_run(run_name=mlflow_run_name)
                else:
                    run_context = nullcontext()

                # Convert data to DataFrames
                train_df = pd.DataFrame(train_data)
                test_df = pd.DataFrame(test_data)
                calib_df = pd.DataFrame(calib_data)

                with run_context as run:
                    # Initialize H2O
                    h2o.init()

                    # Create H2OFrames
                    train_h2o = h2o.H2OFrame(train_df)
                    test_h2o = h2o.H2OFrame(test_df)
                    calib_h2o = h2o.H2OFrame(calib_df)

                    # Convert target variable to categorical if it's binary
                    # Check if target has only 2 unique values by converting to pandas first
                    target_values = train_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                    if len(set(target_values)) == 2:
                        train_h2o[target_variable] = train_h2o[target_variable].asfactor()
                        test_h2o[target_variable] = test_h2o[target_variable].asfactor()
                        calib_h2o[target_variable] = calib_h2o[target_variable].asfactor()

                    best_params = None
                    optimization_results = None

                    # Optuna optimization if enabled
                    if enable_optuna:
                        def objective(trial):
                            # Suggest hyperparameters
                            max_runtime_secs = trial.suggest_int('max_runtime_secs', 60, 600)
                            max_models = trial.suggest_int('max_models', 10, 100)
                            nfolds = trial.suggest_int('nfolds', 3, 10)
                            balance_classes = trial.suggest_categorical('balance_classes', [True, False])
                            
                            # Create AutoML instance
                            aml = H2OAutoML(
                                max_runtime_secs=max_runtime_secs,
                                max_models=max_models,
                                nfolds=nfolds,
                                balance_classes=balance_classes,
                                seed=42,
                                sort_metric="AUTO"
                            )
                            
                            # Train
                            aml.train(x=feature_columns, y=target_variable, training_frame=train_h2o)
                            
                            # Evaluate on validation (using cross-validation results)
                            if hasattr(aml.leader, 'auc'):
                                return aml.leader.auc()[0][0]  # Cross-validation AUC
                            else:
                                return aml.leader.rmse()[0][0]  # For regression, minimize RMSE
                        
                        # Run optimization
                        study = optuna.create_study(
                            direction='maximize',
                            sampler=TPESampler(seed=42)
                        )
                        
                        study.optimize(objective, n_trials=optuna_n_trials, timeout=optuna_timeout)
                        
                        best_params = study.best_params
                        optimization_results = {
                            'best_params': best_params,
                            'best_value': study.best_value,
                            'n_trials': len(study.trials)
                        }
                    
                    # Train final model with best parameters or defaults
                    final_params = best_params if best_params else {}
                    
                    aml = H2OAutoML(
                        max_runtime_secs=final_params.get('max_runtime_secs', 300),
                        max_models=final_params.get('max_models', 20),
                        nfolds=final_params.get('nfolds', 5),
                        balance_classes=final_params.get('balance_classes', True),
                        seed=42,
                        sort_metric="AUTO"
                    )
                    
                    # Train final model
                    aml.train(x=feature_columns, y=target_variable, training_frame=train_h2o)
                    
                    # Evaluate on test set
                    test_perf = aml.leader.model_performance(test_h2o)
                    test_metrics = {}
                    
                    # Handle classification metrics
                    try:
                        if hasattr(test_perf, 'auc'):
                            auc_value = test_perf.auc()
                            test_metrics['auc'] = auc_value[0][0] if hasattr(auc_value, '__getitem__') else auc_value
                    except:
                        pass
                        
                    try:
                        if hasattr(test_perf, 'logloss'):
                            logloss_value = test_perf.logloss()
                            test_metrics['logloss'] = logloss_value[0][0] if hasattr(logloss_value, '__getitem__') else logloss_value
                    except:
                        pass
                        
                    # Calculate Brier Score for binary classification
                    try:
                        if len(set(target_values)) == 2:  # Binary classification
                            # Get predicted probabilities
                            test_pred = aml.leader.predict(test_h2o)
                            test_probs = test_pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()  # Probability of positive class
                            test_actual = test_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                            
                            # Convert to numeric if categorical
                            if test_actual.dtype == 'object':
                                test_actual = (test_actual == test_actual[0]).astype(int)
                            
                            # Calculate Brier Score
                            brier_score = np.mean((test_probs - test_actual) ** 2)
                            test_metrics['brier_score'] = brier_score
                    except Exception as e:
                        print(f"Could not calculate Brier score: {e}")
                        
                    # Handle regression metrics
                    try:
                        if hasattr(test_perf, 'rmse'):
                            rmse_value = test_perf.rmse()
                            test_metrics['rmse'] = rmse_value[0][0] if hasattr(rmse_value, '__getitem__') else rmse_value
                    except:
                        pass
                        
                    try:
                        if hasattr(test_perf, 'mae'):
                            mae_value = test_perf.mae()
                            test_metrics['mae'] = mae_value[0][0] if hasattr(mae_value, '__getitem__') else mae_value
                    except:
                        pass
                    
                    # Evaluate on calibration set
                    calib_perf = aml.leader.model_performance(calib_h2o)
                    calib_metrics = {}
                    
                    # Handle classification metrics
                    try:
                        if hasattr(calib_perf, 'auc'):
                            auc_value = calib_perf.auc()
                            calib_metrics['auc'] = auc_value[0][0] if hasattr(auc_value, '__getitem__') else auc_value
                    except:
                        pass
                        
                    try:
                        if hasattr(calib_perf, 'logloss'):
                            logloss_value = calib_perf.logloss()
                            calib_metrics['logloss'] = logloss_value[0][0] if hasattr(logloss_value, '__getitem__') else logloss_value
                    except:
                        pass
                        
                    # Calculate Brier Score for calibration set
                    try:
                        if len(set(target_values)) == 2:  # Binary classification
                            # Get predicted probabilities
                            calib_pred = aml.leader.predict(calib_h2o)
                            calib_probs = calib_pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()  # Probability of positive class
                            calib_actual = calib_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                            
                            # Convert to numeric if categorical
                            if calib_actual.dtype == 'object':
                                calib_actual = (calib_actual == calib_actual[0]).astype(int)
                            
                            # Calculate Brier Score
                            brier_score = np.mean((calib_probs - calib_actual) ** 2)
                            calib_metrics['brier_score'] = brier_score
                    except Exception as e:
                        print(f"Could not calculate Brier score for calibration set: {e}")
                        
                    # Handle regression metrics
                    try:
                        if hasattr(calib_perf, 'rmse'):
                            rmse_value = calib_perf.rmse()
                            calib_metrics['rmse'] = rmse_value[0][0] if hasattr(rmse_value, '__getitem__') else rmse_value
                    except:
                        pass
                        
                    try:
                        if hasattr(calib_perf, 'mae'):
                            mae_value = calib_perf.mae()
                            calib_metrics['mae'] = mae_value[0][0] if hasattr(mae_value, '__getitem__') else mae_value
                    except:
                        pass

                    # Save model if directory provided
                    model_path = None
                    if model_directory or log_path:
                        save_path = model_directory if model_directory else log_path
                        model_path = h2o.save_model(model=aml.leader, path=save_path, force=True)

                    # Get leaderboard
                    leaderboard_df = aml.leaderboard.as_data_frame(use_multi_thread=True)
                    # Compute Brier Score for all models in the leaderboard (binary classification only)
                    if len(set(target_values)) == 2:
                        brier_scores = []
                        for model_id in leaderboard_df['model_id']:
                            model = h2o.get_model(model_id)
                            pred = model.predict(test_h2o)
                            if 'p1' in pred.columns:
                                probs = pred['p1'].as_data_frame(use_multi_thread=True).values.flatten()
                            else:
                                # fallback to first probability column if p1 not present
                                prob_cols = [col for col in pred.columns if col.startswith('p')]
                                probs = pred[prob_cols[0]].as_data_frame(use_multi_thread=True).values.flatten()
                            actual = test_h2o[target_variable].as_data_frame(use_multi_thread=True).values.flatten()
                            if actual.dtype == 'object':
                                actual = (actual == actual[0]).astype(int)
                            brier = np.mean((probs - actual) ** 2)
                            brier_scores.append(brier)
                        leaderboard_df['brier_score'] = brier_scores
                        leaderboard_df = leaderboard_df.sort_values('brier_score', ascending=True).reset_index(drop=True)
                    leaderboard_dict = leaderboard_df.to_dict()

                    # Set best model as the one with the lowest Brier Score (if available)
                    if 'brier_score' in leaderboard_df.columns:
                        best_model_id = leaderboard_df.loc[0, 'model_id']
                        best_model = h2o.get_model(best_model_id)
                        if model_directory or log_path:
                            save_path = model_directory if model_directory else log_path
                            model_path = h2o.save_model(model=best_model, path=save_path, force=True)
                        else:
                            model_path = None
                    else:
                        best_model_id = aml.leader.model_id
                        model_path = model_path if 'model_path' in locals() else None

                    # Get model type and parameters
                    model_type = type(best_model).__name__
                    model_params = best_model.params if hasattr(best_model, "params") else {}

                    # Log to MLflow if enabled
                    if enable_mlflow and run:
                        mlflow.log_metrics(test_metrics)
                        mlflow.log_metrics({f"calib_{k}": v for k, v in calib_metrics.items()})
                        if optimization_results:
                            mlflow.log_params(best_params)
                            mlflow.log_metric("optuna_best_value", optimization_results['best_value'])
                        mlflow.h2o.log_model(aml.leader, artifact_path="model")

                    # Prepare results
                    results = {
                        'leaderboard': leaderboard_dict,
                        'best_model_id': best_model_id,
                        'model_path': model_path,
                        'test_metrics': test_metrics,
                        'calibration_metrics': calib_metrics,
                        'optimization_results': optimization_results,
                        'model_results': {
                            'model_flavor': 'H2O AutoML Enhanced',
                            'model_path': model_path,
                            'best_model_id': best_model_id,
                            'test_performance': test_metrics,
                            'calibration_performance': calib_metrics
                        },
                        'model_type': model_type,
                        'model_parameters': model_params,
                        'custom_models': {},  # Always include this key in fallback
                    }

                    return results
            ```
            
            CRITICAL REQUIREMENTS:
            1. Follow the EXACT template structure above
            2. Use consistent 4-space indentation
            3. Ensure all code blocks are properly closed
            4. Do not add any extra code outside the function
            5. Make sure all conditional blocks are complete
            6. Return ONLY the Python function code
            """,
            input_variables=[
                "user_instructions",
                "function_name",
                "target_variable",
                "feature_columns",
                "recommended_steps",
                "all_datasets_summary",
                "model_directory",
                "log_path",
                "enable_optuna",
                "optuna_n_trials",
                "optuna_timeout",
                "enable_mlflow",
            ],
        )

        recommended_steps = state.get("recommended_steps", "")
        print(f"\nDEBUG: recommended_steps type: {type(recommended_steps)}")
        print(f"DEBUG: recommended_steps value: {recommended_steps}")
        print(f"DEBUG: state keys: {state.keys()}")

        # More thorough type checking and conversion
        if isinstance(recommended_steps, dict):
            print(f"DEBUG: Dict keys present: {recommended_steps.keys()}")
            if "template" in recommended_steps:
                print("DEBUG: Found template key, extracting template value")
                recommended_steps = recommended_steps["template"]
            else:
                print("DEBUG: No template key found, converting entire dict to string")
                recommended_steps = str(recommended_steps)
        elif hasattr(recommended_steps, "template"):
            print("DEBUG: Object has template attribute, extracting template")
            recommended_steps = recommended_steps.template
        elif not isinstance(recommended_steps, str):
            print(f"DEBUG: Converting type {type(recommended_steps)} to string")
            recommended_steps = str(recommended_steps)

        print(f"DEBUG: Final recommended_steps type: {type(recommended_steps)}")
        print(f"DEBUG: Final recommended_steps value: {recommended_steps}")

        h2o_code_agent = code_prompt | llm | PythonOutputParser()

        resp = h2o_code_agent.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "function_name": function_name,
                "target_variable": state.get("target_variable"),
                "feature_columns": str(state.get("feature_columns")),
                "recommended_steps": recommended_steps,
                "all_datasets_summary": all_datasets_summary_str,
                "model_directory": model_directory,
                "log_path": log_path,
                "enable_optuna": enable_optuna,
                "optuna_n_trials": optuna_n_trials,
                "optuna_timeout": optuna_timeout,
                "enable_mlflow": enable_mlflow,
            }
        )

        resp = relocate_imports_inside_function(resp)
        resp = add_comments_to_top(resp, agent_name=AGENT_NAME)

        # Fix indentation issues
        resp = fix_code_indentation(resp)

        # Validate and fix any remaining syntax issues
        resp = validate_and_fix_code(resp)

        # Log the code snippet if requested
        file_path, f_name = log_ai_function(
            response=resp,
            file_name=file_name,
            log=log,
            log_path=log_path,
            overwrite=overwrite,
        )

        return {
            "h2o_train_function": resp,
            "h2o_train_function_path": file_path,
            "h2o_train_file_name": f_name,
            "h2o_train_function_name": function_name,
        }

    # Human Review node
    prompt_text_human_review = "Are the following Enhanced ML instructions with Optuna optimization correct? (Answer 'yes' or provide modifications)\n{recommended_steps}"

    if not bypass_explain_code:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["recommend_ml_steps", "explain_h2o_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="explain_h2o_code",
                no_goto="recommend_ml_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="h2o_train_function",
            )

    else:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["recommend_ml_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="recommend_ml_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="h2o_train_function",
            )

    # Explain H2O code node
    def explain_h2o_code(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * EXPLAINING ENHANCED H2O AUTOML CODE")

        # Get the generated code
        code = state.get("h2o_train_function", "")

        if code:
            print("Enhanced H2O AutoML Code with Optuna Optimization:")
            print(
                "This Enhanced H2O AutoML function uses separate train/test/calibration datasets and optional Optuna hyperparameter optimization for superior model performance."
            )
            print(
                f"\nGenerated function: {state.get('h2o_train_function_name', 'h2o_automl_enhanced')}"
            )
            print(f"Code saved to: {state.get('h2o_train_function_path', 'N/A')}")

        return {
            "messages": [
                AIMessage(
                    content="H2O AutoML code explanation completed", role="H2OAgent"
                )
            ]
        }

    # Execute H2O code node
    def execute_h2o_code(state: GraphState):
        print("    * EXECUTING GENERATED CODE")

        try:
            # Get the generated code - check for H2O specific keys first
            code = state.get("h2o_train_function", "")
            if not code:
                return {"h2o_train_error": "No code found to execute"}

            # Get function name - check for H2O specific key first
            function_name = state.get("h2o_train_function_name", "h2o_automl_enhanced")
            if not function_name:
                function_name = "h2o_automl_enhanced"

            # Get data - handle H2O agent's data structure
            train_data = state.get("train_data", {})
            test_data = state.get("test_data", {})
            calib_data = state.get("calib_data", {})
            target_variable = state.get("target_variable", "")
            feature_columns = state.get("feature_columns", [])

            # Create a local namespace for execution
            local_namespace = {}

            # Execute the code to define the function
            exec(code, globals(), local_namespace)

            # Get the function from the namespace
            if function_name not in local_namespace:
                return {
                    "h2o_train_error": f"Function '{function_name}' not found in generated code"
                }

            func = local_namespace[function_name]

            # Prepare arguments for the H2O function
            args = {
                "train_data": train_data,
                "test_data": test_data,
                "calib_data": calib_data,
                "target_variable": target_variable,
                "feature_columns": feature_columns,
            }

            # Execute the function
            print(f"Executing function '{function_name}' with {len(args)} arguments...")
            result = func(**args)

            # Extract results from the function output
            if isinstance(result, dict):
                # Get the H2O leaderboard from the result
                h2o_leaderboard = result.get("leaderboard")

                # Convert data back to DataFrames for unified leaderboard creation
                import json

                import h2o
                import numpy as np
                import pandas as pd

                X_train_df = pd.DataFrame(train_data)
                y_train_series = X_train_df[target_variable]
                X_train_df = X_train_df.drop(columns=[target_variable])

                X_test_df = pd.DataFrame(test_data)
                y_test_series = X_test_df[target_variable]
                X_test_df = X_test_df.drop(columns=[target_variable])

                X_calib_df = pd.DataFrame(calib_data)
                y_calib_series = X_calib_df[target_variable]
                X_calib_df = X_calib_df.drop(columns=[target_variable])

                # Create H2O frame for test data
                test_h2o = h2o.H2OFrame(X_test_df)
                test_h2o[target_variable] = h2o.H2OFrame(y_test_series.to_frame())

                # Convert H2O leaderboard to DataFrame if it's a dict
                if isinstance(h2o_leaderboard, dict):
                    h2o_leaderboard_df = pd.DataFrame(h2o_leaderboard)
                else:
                    h2o_leaderboard_df = h2o_leaderboard

                # Create unified leaderboard with custom models (standalone function)
                def create_unified_leaderboard(
                    h2o_leaderboard_df,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    target_variable,
                ):
                    """Create unified leaderboard with H2O and custom models."""
                    import optuna
                    from catboost import CatBoostClassifier
                    from lightgbm import LGBMClassifier
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.metrics import (
                        brier_score_loss,
                        log_loss,
                        roc_auc_score,
                    )
                    from xgboost import XGBClassifier

                    custom_models = {}
                    custom_results = []

                    # Helper for Optuna tuning
                    def tune_model(objective, n_trials=10, timeout=60):
                        study = optuna.create_study(direction="maximize")
                        study.optimize(objective, n_trials=n_trials, timeout=timeout)
                        return study.best_trial.params

                    # Train custom models with Optuna optimization
                    try:
                        # Random Forest
                        def rf_objective(trial):
                            params = {
                                "n_estimators": trial.suggest_int(
                                    "n_estimators", 50, 200
                                ),
                                "max_depth": trial.suggest_int("max_depth", 2, 15),
                                "min_samples_split": trial.suggest_int(
                                    "min_samples_split", 2, 10
                                ),
                            }
                            clf = RandomForestClassifier(**params, random_state=42)
                            clf.fit(X_train, y_train)
                            probas = clf.predict_proba(X_test)[:, 1]
                            return roc_auc_score(y_test, probas)

                        rf_params = tune_model(rf_objective)
                        rf = RandomForestClassifier(**rf_params, random_state=42)
                        rf.fit(X_train, y_train)
                        custom_models["Random Forest"] = rf

                        # XGBoost
                        def xgb_objective(trial):
                            params = {
                                "n_estimators": trial.suggest_int(
                                    "n_estimators", 50, 200
                                ),
                                "max_depth": trial.suggest_int("max_depth", 2, 15),
                                "learning_rate": trial.suggest_float(
                                    "learning_rate", 0.01, 0.3
                                ),
                                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                                "colsample_bytree": trial.suggest_float(
                                    "colsample_bytree", 0.5, 1.0
                                ),
                                "use_label_encoder": False,
                                "eval_metric": "logloss",
                            }
                            clf = XGBClassifier(**params, random_state=42)
                            clf.fit(X_train, y_train)
                            probas = clf.predict_proba(X_test)[:, 1]
                            return roc_auc_score(y_test, probas)

                        xgb_params = tune_model(xgb_objective)
                        xgb = XGBClassifier(**xgb_params, random_state=42)
                        xgb.fit(X_train, y_train)
                        custom_models["XGBoost"] = xgb

                        # CatBoost
                        def cat_objective(trial):
                            params = {
                                "iterations": trial.suggest_int("iterations", 50, 200),
                                "depth": trial.suggest_int("depth", 2, 10),
                                "learning_rate": trial.suggest_float(
                                    "learning_rate", 0.01, 0.3
                                ),
                                "verbose": 0,
                            }
                            clf = CatBoostClassifier(**params, random_state=42)
                            clf.fit(X_train, y_train)
                            probas = clf.predict_proba(X_test)[:, 1]
                            return roc_auc_score(y_test, probas)

                        cat_params = tune_model(cat_objective)
                        cat = CatBoostClassifier(
                            **cat_params, random_state=42, verbose=0
                        )
                        cat.fit(X_train, y_train)
                        custom_models["CatBoost"] = cat

                        # LightGBM
                        def lgbm_objective(trial):
                            params = {
                                "n_estimators": trial.suggest_int(
                                    "n_estimators", 50, 200
                                ),
                                "max_depth": trial.suggest_int("max_depth", 2, 15),
                                "learning_rate": trial.suggest_float(
                                    "learning_rate", 0.01, 0.3
                                ),
                                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                                "colsample_bytree": trial.suggest_float(
                                    "colsample_bytree", 0.5, 1.0
                                ),
                            }
                            clf = LGBMClassifier(**params, random_state=42)
                            clf.fit(X_train, y_train)
                            probas = clf.predict_proba(X_test)[:, 1]
                            return roc_auc_score(y_test, probas)

                        lgbm_params = tune_model(lgbm_objective)
                        lgbm = LGBMClassifier(**lgbm_params, random_state=42)
                        lgbm.fit(X_train, y_train)
                        custom_models["LightGBM"] = lgbm

                        # Logistic Regression L1
                        def lr_l1_objective(trial):
                            params = {
                                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                                "penalty": "l1",
                                "solver": "liblinear",
                            }
                            clf = LogisticRegression(
                                **params, random_state=42, max_iter=1000
                            )
                            clf.fit(X_train, y_train)
                            probas = clf.predict_proba(X_test)[:, 1]
                            return roc_auc_score(y_test, probas)

                        lr_l1_params = tune_model(lr_l1_objective)
                        lr_l1 = LogisticRegression(
                            **lr_l1_params, random_state=42, max_iter=1000
                        )
                        lr_l1.fit(X_train, y_train)
                        custom_models["Logistic Regression (L1)"] = lr_l1

                        # Logistic Regression L2
                        def lr_l2_objective(trial):
                            params = {
                                "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                                "penalty": "l2",
                                "solver": "liblinear",
                            }
                            clf = LogisticRegression(
                                **params, random_state=42, max_iter=1000
                            )
                            clf.fit(X_train, y_train)
                            probas = clf.predict_proba(X_test)[:, 1]
                            return roc_auc_score(y_test, probas)

                        lr_l2_params = tune_model(lr_l2_objective)
                        lr_l2 = LogisticRegression(
                            **lr_l2_params, random_state=42, max_iter=1000
                        )
                        lr_l2.fit(X_train, y_train)
                        custom_models["Logistic Regression (L2)"] = lr_l2

                    except Exception as e:
                        print(f"Warning: Could not train all custom models: {e}")

                    # Collect results for custom models
                    for name, model in custom_models.items():
                        try:
                            probas = model.predict_proba(X_test)[:, 1]
                            auc = roc_auc_score(y_test, probas)
                            brier = brier_score_loss(y_test, probas)
                            logloss = log_loss(y_test, probas)
                            custom_results.append(
                                {
                                    "model_id": name,
                                    "auc": auc,
                                    "brier_score": brier,
                                    "logloss": logloss,
                                    "probas": probas,
                                }
                            )
                        except Exception as e:
                            print(f"Warning: Could not evaluate {name}: {e}")

                    # Prepare unified leaderboard
                    h2o_cols = set(h2o_leaderboard_df.columns)
                    custom_cols = set()
                    for res in custom_results:
                        custom_cols.update(res.keys())
                    custom_cols.discard("probas")
                    all_cols = list(
                        {"model_id"} | (h2o_cols | custom_cols) - {"probas"}
                    )
                    if "model_parameters" not in all_cols:
                        all_cols.append("model_parameters")

                    # Build the leaderboard with all columns
                    leaderboard = h2o_leaderboard_df.copy()
                    leaderboard = leaderboard.reindex(columns=all_cols)

                    # Add custom models to leaderboard
                    for res in custom_results:
                        row = {
                            col: res.get(col, np.nan)
                            for col in all_cols
                            if col != "model_parameters"
                        }
                        # Add model parameters for custom models
                        model_obj = custom_models.get(res["model_id"])
                        if model_obj is not None:
                            row["model_parameters"] = json.dumps(model_obj.get_params())
                        else:
                            row["model_parameters"] = np.nan
                        leaderboard = pd.concat(
                            [leaderboard, pd.DataFrame([row])], ignore_index=True
                        )

                    # Add model parameters for H2O models
                    for idx, model_id in enumerate(h2o_leaderboard_df["model_id"]):
                        try:
                            model = h2o.get_model(model_id)
                            leaderboard.at[idx, "model_parameters"] = json.dumps(
                                model.params
                            )
                        except Exception:
                            leaderboard.at[idx, "model_parameters"] = np.nan

                    # Sort by brier_score or auc
                    if "brier_score" in leaderboard.columns:
                        leaderboard = leaderboard.sort_values(
                            "brier_score", ascending=True
                        ).reset_index(drop=True)
                    elif "auc" in leaderboard.columns:
                        leaderboard = leaderboard.sort_values(
                            "auc", ascending=False
                        ).reset_index(drop=True)

                    return leaderboard, custom_models

                # Create unified leaderboard
                unified_leaderboard, custom_models = create_unified_leaderboard(
                    h2o_leaderboard_df=h2o_leaderboard_df,
                    X_train=X_train_df,
                    y_train=y_train_series,
                    X_test=X_test_df,
                    y_test=y_test_series,
                    target_variable=target_variable,
                )

                # Convert unified leaderboard to dict for response
                unified_leaderboard_dict = (
                    unified_leaderboard.to_dict()
                    if hasattr(unified_leaderboard, "to_dict")
                    else unified_leaderboard
                )

                return {
                    "leaderboard": unified_leaderboard_dict,  # Use unified leaderboard instead of H2O only
                    "best_model_id": result.get("best_model_id"),
                    "model_path": result.get("model_path"),
                    "model_results": result.get("model_results", {}),
                    "optimization_results": result.get("optimization_results"),
                    "test_metrics": result.get("test_metrics"),
                    "calibration_metrics": result.get("calibration_metrics"),
                    "custom_models": custom_models,  # Add custom models to response
                    "result": result,
                    "h2o_train_error": None,  # Clear any previous errors
                }
            else:
                return {"result": result, "h2o_train_error": None}

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            print(f"❌ {error_msg}")
            return {"h2o_train_error": error_msg}

    # Fix H2O code node
    def fix_h2o_code(state: GraphState):
        print(format_agent_name(AGENT_NAME))
        print("    * FIXING H2O AUTOML CODE")

        # Get the current error and code
        error = state.get("h2o_train_error", "")
        code = state.get("h2o_train_function", "")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        print(f"Attempt {retry_count + 1} of {max_retries}")
        print(f"Error: {error}")

        # For now, just return the same code and increment retry count
        # In a real implementation, you would use an LLM to fix the code
        return {
            "h2o_train_function": code,
            "retry_count": retry_count + 1,
            "h2o_train_error": None,
        }

    # Route function for error handling
    def route_h2o_code(state: GraphState) -> Literal["fix_h2o_code", "__end__"]:
        error = state.get("h2o_train_error")
        if error:
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 3)
            if retry_count < max_retries:
                return "fix_h2o_code"
        return "__end__"

    # Create the graph
    if human_in_the_loop:
        if not bypass_explain_code:
            # Create a proper StateGraph for the H2O agent
            from langgraph.graph import END, StateGraph

            workflow = StateGraph(GraphState)

            # Add nodes
            workflow.add_node("recommend_ml_steps", recommend_ml_steps)
            workflow.add_node("human_review", human_review)
            workflow.add_node("explain_h2o_code", explain_h2o_code)
            workflow.add_node("create_h2o_code", create_h2o_code)
            workflow.add_node("execute_h2o_code", execute_h2o_code)
            workflow.add_node("fix_h2o_code", fix_h2o_code)

            # Set entry point
            workflow.set_entry_point("recommend_ml_steps")

            # Add edges
            workflow.add_edge("recommend_ml_steps", "human_review")
            workflow.add_edge("human_review", "explain_h2o_code")
            workflow.add_edge("explain_h2o_code", "create_h2o_code")
            workflow.add_edge("create_h2o_code", "execute_h2o_code")

            # Add conditional edges for error handling
            workflow.add_conditional_edges(
                "execute_h2o_code",
                route_h2o_code,
                {"fix_h2o_code": "fix_h2o_code", "__end__": END},
            )
            workflow.add_edge("fix_h2o_code", "execute_h2o_code")

            graph = workflow.compile(checkpointer=checkpointer)
        else:
            # Create a proper StateGraph for the H2O agent (without explain step)
            from langgraph.graph import END, StateGraph

            workflow = StateGraph(GraphState)

            # Add nodes
            workflow.add_node("recommend_ml_steps", recommend_ml_steps)
            workflow.add_node("human_review", human_review)
            workflow.add_node("create_h2o_code", create_h2o_code)
            workflow.add_node("execute_h2o_code", execute_h2o_code)
            workflow.add_node("fix_h2o_code", fix_h2o_code)

            # Set entry point
            workflow.set_entry_point("recommend_ml_steps")

            # Add edges
            workflow.add_edge("recommend_ml_steps", "human_review")
            workflow.add_edge("human_review", "create_h2o_code")
            workflow.add_edge("create_h2o_code", "execute_h2o_code")

            # Add conditional edges for error handling
            workflow.add_conditional_edges(
                "execute_h2o_code",
                route_h2o_code,
                {"fix_h2o_code": "fix_h2o_code", "__end__": END},
            )
            workflow.add_edge("fix_h2o_code", "execute_h2o_code")

            graph = workflow.compile(checkpointer=checkpointer)
    else:
        if not bypass_explain_code:
            # Create a proper StateGraph for the H2O agent (without human review)
            from langgraph.graph import END, StateGraph

            workflow = StateGraph(GraphState)

            # Add nodes
            workflow.add_node("recommend_ml_steps", recommend_ml_steps)
            workflow.add_node("explain_h2o_code", explain_h2o_code)
            workflow.add_node("create_h2o_code", create_h2o_code)
            workflow.add_node("execute_h2o_code", execute_h2o_code)
            workflow.add_node("fix_h2o_code", fix_h2o_code)

            # Set entry point
            workflow.set_entry_point("recommend_ml_steps")

            # Add edges
            workflow.add_edge("recommend_ml_steps", "explain_h2o_code")
            workflow.add_edge("explain_h2o_code", "create_h2o_code")
            workflow.add_edge("create_h2o_code", "execute_h2o_code")

            # Add conditional edges for error handling
            workflow.add_conditional_edges(
                "execute_h2o_code",
                route_h2o_code,
                {"fix_h2o_code": "fix_h2o_code", "__end__": END},
            )
            workflow.add_edge("fix_h2o_code", "execute_h2o_code")

            graph = workflow.compile(checkpointer=checkpointer)
        else:
            # Create a proper StateGraph for the H2O agent (minimal workflow)
            from langgraph.graph import END, StateGraph

            workflow = StateGraph(GraphState)

            # Add nodes
            workflow.add_node("recommend_ml_steps", recommend_ml_steps)
            workflow.add_node("create_h2o_code", create_h2o_code)
            workflow.add_node("execute_h2o_code", execute_h2o_code)
            workflow.add_node("fix_h2o_code", fix_h2o_code)

            # Set entry point
            workflow.set_entry_point("recommend_ml_steps")

            # Add edges
            workflow.add_edge("recommend_ml_steps", "create_h2o_code")
            workflow.add_edge("create_h2o_code", "execute_h2o_code")

            # Add conditional edges for error handling
            workflow.add_conditional_edges(
                "execute_h2o_code",
                route_h2o_code,
                {"fix_h2o_code": "fix_h2o_code", "__end__": END},
            )
            workflow.add_edge("fix_h2o_code", "execute_h2o_code")

            graph = workflow.compile(checkpointer=checkpointer)

    return graph
