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
            print(f"Could not calculate Brier score: {{e}}")
            
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
            print(f"Could not calculate Brier score for calibration set: {{e}}")
            
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
            'model_parameters': model_params
        }

        # Add CDF vs EDF PDF for all models
        try:
            cdf_edf_pdf_path = plot_cdf_vs_edf_for_models(leaderboard_df, test_h2o, target_variable, logs_dir=log_path or "logs/")
            results['cdf_edf_pdf_path'] = cdf_edf_pdf_path
        except Exception as e:
            print(f"Could not generate CDF vs EDF PDF: {e}")
            results['cdf_edf_pdf_path'] = None

        return results
"""


AGENT_NAME = "h2o_ml_agent_enhanced"
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

        response = self._compiled_graph.invoke(initial_state, **kwargs)
        self.response = response

        # Automatically generate CDF vs EDF PDF after training
        leaderboard_df = self.get_leaderboard()
        # Try to get test_h2o and target_variable from arguments or state
        test_h2o = kwargs.get("test_h2o", None)
        if test_h2o is None:
            try:
                import h2o

                test_h2o = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
            except Exception as e:
                print(f"Could not create H2OFrame for test set: {e}")
                test_h2o = None
        target_variable = y_train.name
        if (
            leaderboard_df is not None
            and test_h2o is not None
            and target_variable is not None
        ):
            pdf_path = self.plot_reliability_curves_for_leaderboard(
                leaderboard_df,
                test_h2o,
                target_variable,
                logs_dir=self._params.get("log_path", "logs/"),
            )
            print(f"Reliability curves saved to: {pdf_path}")
        return None

    def get_leaderboard(self):
        """Returns the H2O AutoML leaderboard as a DataFrame."""
        if self.response and "leaderboard" in self.response:
            return pd.DataFrame(self.response["leaderboard"])
        return None

    def get_best_model_id(self):
        """Returns the best model ID from the AutoML run."""
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

    def plot_reliability_curves_for_leaderboard(
        self, leaderboard_df, test_h2o, target_variable, logs_dir="logs/"
    ):
        import os

        import h2o
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.calibration import calibration_curve

        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        pdf_path = os.path.join(logs_dir, "reliability_curves_leaderboard.pdf")
        plt.figure(figsize=(15, 5))
        for idx, model_id in enumerate(
            leaderboard_df["model_id"][:3]
        ):  # Plot for top 3 models
            model = h2o.get_model(model_id)
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
            # Convert to numeric if categorical
            if hasattr(actual, "dtype") and actual.dtype == "object":
                actual = (actual == actual[0]).astype(int)
            else:
                actual = actual.astype(int)
            prob_true, prob_pred = calibration_curve(actual, probs, n_bins=10)
            plt.subplot(1, 3, idx + 1)
            plt.plot(prob_pred, prob_true, marker="o", label="Test")
            plt.plot([0, 1], [0, 1], "r--", label="Perfectly Calibrated")
            plt.xlabel("Mean Predicted Probability")
            plt.ylabel("Fraction of Positives")
            plt.title(model_id[:30])
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.legend()
        plt.suptitle("Reliability Curves (Calibration) for Top 3 Models")
        plt.tight_layout()
        plt.savefig(pdf_path)
        plt.close()
        return pdf_path


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
                        optimization_results = {{
                            'best_params': best_params,
                            'best_value': study.best_value,
                            'n_trials': len(study.trials)
                        }}
                    
                    # Train final model with best parameters or defaults
                    final_params = best_params if best_params else {{}}
                    
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
                    test_metrics = {{}}
                    
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
                        print(f"Could not calculate Brier score: {{e}}")
                        
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
                    calib_metrics = {{}}
                    
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
                        print(f"Could not calculate Brier score for calibration set: {{e}}")
                        
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

                    # Log to MLflow if enabled
                    if enable_mlflow and run:
                        mlflow.log_metrics(test_metrics)
                        mlflow.log_metrics({{f"calib_{{k}}": v for k, v in calib_metrics.items()}})
                        if optimization_results:
                            mlflow.log_params(best_params)
                            mlflow.log_metric("optuna_best_value", optimization_results['best_value'])
                        mlflow.h2o.log_model(aml.leader, artifact_path="model")

                    # Prepare results
                    results = {{
                        'leaderboard': leaderboard_dict,
                        'best_model_id': best_model_id,
                        'model_path': model_path,
                        'test_metrics': test_metrics,
                        'calibration_metrics': calib_metrics,
                        'optimization_results': optimization_results,
                        'model_results': {{
                            'model_flavor': 'H2O AutoML Enhanced',
                            'model_path': model_path,
                            'best_model_id': best_model_id,
                            'test_performance': test_metrics,
                            'calibration_performance': calib_metrics
                        }},
                        'model_type': model_type,
                        'model_parameters': model_params
                    }}

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
                return {
                    "leaderboard": result.get("leaderboard"),
                    "best_model_id": result.get("best_model_id"),
                    "model_path": result.get("model_path"),
                    "model_results": result.get("model_results", {}),
                    "optimization_results": result.get("optimization_results"),
                    "test_metrics": result.get("test_metrics"),
                    "calibration_metrics": result.get("calibration_metrics"),
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
