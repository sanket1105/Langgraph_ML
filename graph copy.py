import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.types import StreamMode

from parsers import PythonOutputParser
from regex import format_agent_name, remove_consecutive_duplicates


class BaseAgent:
    """
    Base class for all agents in the system.
    Provides common functionality for agent initialization and execution.
    """

    def __init__(self, **params):
        self._params = params
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Override this method in subclasses to create the specific graph."""
        raise NotImplementedError("Subclasses must implement _make_compiled_graph")

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs
    ) -> Any:
        """Invoke the agent with input data."""
        return self._compiled_graph.invoke(input=input, config=config, **kwargs)

    def stream(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs
    ) -> Any:
        """Stream the agent execution."""
        return self._compiled_graph.stream(input=input, config=config, **kwargs)


def create_coding_agent_graph(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="generated_code.py",
    function_name="main_function",
    model_directory=None,
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    enable_mlflow=False,
    mlflow_tracking_uri=None,
    mlflow_experiment_name="Default Experiment",
    mlflow_run_name=None,
    enable_optuna=False,
    optuna_n_trials=50,
    optuna_timeout=300,
    checkpointer=None,
):
    """
    Create a coding agent graph for general code generation tasks.
    This is a simplified version that can be extended for specific use cases.
    """
    import operator
    from typing import Annotated, Sequence, TypedDict

    from langchain_core.messages import BaseMessage
    from langgraph.graph import StateGraph

    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        code: str
        error: str
        result: dict

    def generate_code(state: GraphState):
        """Generate code based on user instructions."""
        # Placeholder implementation
        return {"code": "# Generated code placeholder", "error": None}

    def execute_code(state: GraphState):
        """Execute the generated code."""
        # Placeholder implementation
        return {"result": {"status": "success"}, "error": None}

    # Create the workflow
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("execute_code", execute_code)

    # Set entry point
    workflow.set_entry_point("generate_code")

    # Add edges
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", END)

    return workflow.compile()


def node_func_execute_agent_code_on_data(state):
    """Execute agent code on data."""
    print("    * EXECUTING GENERATED CODE")

    try:
        # Get the generated code - check for H2O specific keys first
        code = state.get("h2o_train_function", "")
        if not code:
            # Fallback to generic code key
            code = state.get("code", "")
        if not code:
            return {"h2o_train_error": "No code found to execute"}

        # Get function name - check for H2O specific key first
        function_name = state.get("h2o_train_function_name", "h2o_automl_enhanced")
        if not function_name:
            # Fallback to generic function name key
            function_name = state.get("function_name", "main_function")

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


def node_func_fix_agent_code(state):
    """Fix agent code if there are errors."""
    # Placeholder implementation
    return {"code": "# Fixed code", "error": None}


def node_func_human_review(
    state,
    prompt_text=None,
    yes_goto="__end__",
    no_goto="recommend_ml_steps",
    user_instructions_key="user_instructions",
    recommended_steps_key="recommended_steps",
    code_snippet_key="code",
):
    """Human review step with prompt template support."""
    from langchain.prompts import PromptTemplate
    from langchain_core.messages import AIMessage

    # Get the recommended steps from state
    recommended_steps = state.get(recommended_steps_key, "")

    # Create prompt template if prompt_text is provided
    if prompt_text:
        prompt_template = PromptTemplate(
            template=prompt_text, input_variables=["recommended_steps"]
        )

        # Format the prompt
        formatted_prompt = prompt_template.format(recommended_steps=recommended_steps)

        # For now, just return a success message
        # In a real implementation, this would show the prompt to the user and wait for input
        return {
            "messages": [
                AIMessage(
                    content=f"Human review prompt:\n{formatted_prompt}\n\nAutomatically proceeding with 'yes' response.",
                    role="HumanReview",
                )
            ]
        }
    else:
        # Fallback to simple review
        return {
            "messages": [
                AIMessage(
                    content="Human review completed - proceeding with recommended steps",
                    role="HumanReview",
                )
            ]
        }


def node_func_report_agent_outputs(state):
    """Report agent outputs."""
    # Placeholder implementation
    return {"report": "Agent execution completed"}


class ModelingPhase(Enum):
    """Enumeration of modeling workflow phases"""

    DEFINE = "define_model"
    TRAIN = "train_model"
    EVALUATE = "evaluate_model"
    EXPLAIN = "explain_model"
    VALIDATE = "validate_model"
    OPTIMIZE = "optimize_model"


class ModelAgent(CompiledStateGraph):
    """
    Agent for automated model building and evaluation.

    Enhanced workflow with conditional routing and error recovery:
    define model -> train model -> evaluate model -> [validate/optimize] -> explain results
    """

    def __init__(self, **params):
        self._params = params
        self._compiled_graph = self._make_compiled_graph()
        self.response = None
        self.execution_history = []

        # Delegate compiled graph attributes
        attrs = [
            "name",
            "checkpointer",
            "store",
            "output_channels",
            "nodes",
            "stream_mode",
            "builder",
            "channels",
            "input_channels",
            "input_schema",
            "output_schema",
            "debug",
            "interrupt_after_nodes",
            "interrupt_before_nodes",
            "config",
        ]
        for attr in attrs:
            setattr(self, attr, getattr(self._compiled_graph, attr))

    def _make_compiled_graph(self) -> CompiledStateGraph:
        return create_modeling_agent_graph(
            GraphState=self._params.get("GraphState"),
            node_functions=self._params.get("node_functions"),
            define_model_node_name=self._params.get(
                "define_model_node", ModelingPhase.DEFINE.value
            ),
            train_model_node_name=self._params.get(
                "train_model_node", ModelingPhase.TRAIN.value
            ),
            evaluate_model_node_name=self._params.get(
                "evaluate_model_node", ModelingPhase.EVALUATE.value
            ),
            explain_model_node_name=self._params.get(
                "explain_model_node", ModelingPhase.EXPLAIN.value
            ),
            validate_model_node_name=self._params.get(
                "validate_model_node", ModelingPhase.VALIDATE.value
            ),
            optimize_model_node_name=self._params.get(
                "optimize_model_node", ModelingPhase.OPTIMIZE.value
            ),
            error_key=self._params.get("error_key", "error"),
            agent_name=self._params.get("agent_name", "model_agent"),
            enable_optimization=self._params.get("enable_optimization", False),
            max_retries=self._params.get("max_retries", 2),
        )

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs
    ) -> Any:
        """Enhanced invoke with execution tracking"""
        try:
            self.response = self._compiled_graph.invoke(
                input=input, config=config, **kwargs
            )
            self._post_process_response()
            self._track_execution(input, self.response)
            return self.response
        except Exception as e:
            error_response = self._handle_execution_error(e, input)
            return error_response

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        stream_mode: Optional[StreamMode] = None,
        **kwargs,
    ) -> Any:
        """Enhanced stream with execution tracking"""
        try:
            self.response = self._compiled_graph.stream(
                input=input, config=config, stream_mode=stream_mode, **kwargs
            )
            self._post_process_response()
            self._track_execution(input, self.response)
            return self.response
        except Exception as e:
            error_response = self._handle_execution_error(e, input)
            return error_response

    def _post_process_response(self):
        """Post-process response with deduplication and formatting"""
        if self.response and self.response.get("messages"):
            self.response["messages"] = remove_consecutive_duplicates(
                self.response["messages"]
            )

    def _track_execution(self, input_data: Any, output_data: Any):
        """Track execution history for debugging and analysis"""
        execution_record = {
            "timestamp": json.dumps({"input": str(input_data)[:100], "success": True}),
            "input_summary": str(input_data)[:100] if input_data else None,
            "output_summary": str(output_data)[:100] if output_data else None,
            "error": (
                output_data.get("error") if isinstance(output_data, dict) else None
            ),
        }
        self.execution_history.append(execution_record)

    def _handle_execution_error(
        self, error: Exception, input_data: Any
    ) -> Dict[str, Any]:
        """Handle execution errors gracefully"""
        error_response = {
            "error": str(error),
            "messages": [
                AIMessage(
                    content=f"Execution failed: {str(error)}", role="ModelingAgent"
                )
            ],
            "phase": "error_handling",
            "input_data": input_data,
        }
        self._track_execution(input_data, error_response)
        return error_response


# -----------------------------------------------------------------------------
# Enhanced Agent Graph Constructor
# -----------------------------------------------------------------------------


def create_modeling_agent_graph(
    GraphState: Type,
    node_functions: Dict[str, Callable],
    define_model_node_name: str,
    train_model_node_name: str,
    evaluate_model_node_name: str,
    explain_model_node_name: str,
    validate_model_node_name: str,
    optimize_model_node_name: str,
    error_key: str,
    agent_name: str = "model_agent",
    enable_optimization: bool = False,
    max_retries: int = 2,
) -> CompiledStateGraph:
    """
    Constructs an enhanced agent state graph for model building and evaluation.

    Enhanced workflow with conditional routing:
    - define_model: produce model specification or code
    - train_model: train the defined model on data
    - evaluate_model: compute evaluation metrics
    - validate_model: cross-validation and robustness checks
    - optimize_model: hyperparameter tuning (optional)
    - explain_model: generate human-readable summary of results

    Conditional transitions based on performance thresholds and error states.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node(define_model_node_name, node_functions[define_model_node_name])
    workflow.add_node(train_model_node_name, node_functions[train_model_node_name])
    workflow.add_node(
        evaluate_model_node_name, node_functions[evaluate_model_node_name]
    )
    workflow.add_node(explain_model_node_name, node_functions[explain_model_node_name])

    # Optional advanced nodes
    if validate_model_node_name in node_functions:
        workflow.add_node(
            validate_model_node_name, node_functions[validate_model_node_name]
        )

    if enable_optimization and optimize_model_node_name in node_functions:
        workflow.add_node(
            optimize_model_node_name, node_functions[optimize_model_node_name]
        )

    # Entry point
    workflow.set_entry_point(define_model_node_name)

    # Enhanced conditional routing
    def should_optimize(state) -> str:
        """Decide whether to optimize based on performance"""
        if not enable_optimization:
            return explain_model_node_name

        metrics = state.get("metrics", {})
        performance_threshold = state.get("optimization_threshold", 0.8)
        current_performance = metrics.get("primary_metric", 0)

        if current_performance < performance_threshold:
            return optimize_model_node_name
        return explain_model_node_name

    def should_validate(state) -> str:
        """Decide whether additional validation is needed"""
        if validate_model_node_name not in node_functions:
            return should_optimize(state)

        error = state.get(error_key)
        if error:
            return explain_model_node_name  # Skip validation if there are errors

        return validate_model_node_name

    def handle_optimization_result(state) -> str:
        """Route after optimization"""
        optimization_improved = state.get("optimization_improved", False)
        if optimization_improved:
            return evaluate_model_node_name  # Re-evaluate optimized model
        return explain_model_node_name

    # Define edges with conditional routing
    workflow.add_edge(define_model_node_name, train_model_node_name)
    workflow.add_edge(train_model_node_name, evaluate_model_node_name)

    # Conditional edge after evaluation
    if validate_model_node_name in node_functions:
        workflow.add_conditional_edges(
            evaluate_model_node_name,
            should_validate,
            {
                validate_model_node_name: validate_model_node_name,
                optimize_model_node_name: optimize_model_node_name,
                explain_model_node_name: explain_model_node_name,
            },
        )
        workflow.add_conditional_edges(
            validate_model_node_name,
            should_optimize,
            {
                optimize_model_node_name: optimize_model_node_name,
                explain_model_node_name: explain_model_node_name,
            },
        )
    else:
        workflow.add_conditional_edges(
            evaluate_model_node_name,
            should_optimize,
            {
                optimize_model_node_name: optimize_model_node_name,
                explain_model_node_name: explain_model_node_name,
            },
        )

    # Handle optimization loop
    if enable_optimization and optimize_model_node_name in node_functions:
        workflow.add_conditional_edges(
            optimize_model_node_name,
            handle_optimization_result,
            {
                evaluate_model_node_name: evaluate_model_node_name,
                explain_model_node_name: explain_model_node_name,
            },
        )

    # Final edge to END
    workflow.add_edge(explain_model_node_name, END)

    # Compile with enhanced configuration
    return workflow.compile(
        name=agent_name,
        checkpointer=None,  # Add checkpointer if needed for persistence
        interrupt_before=None,  # Add interrupt points if needed for human-in-the-loop
        debug=True,  # Enable debug mode for development
    )


# -----------------------------------------------------------------------------
# Enhanced Node Function Templates
# -----------------------------------------------------------------------------


def node_define_model(
    state: Any,
    prompt: str = None,
    code_key: str = "model_code",
    model_type_key: str = "model_type",
) -> Dict[str, Any]:
    """
    Generate model building code/specification based on user prompt.
    Enhanced with model type detection and validation.
    """
    user_prompt = prompt or state.get("user_prompt", "")

    # Detect model type from prompt
    model_type = _detect_model_type(user_prompt)

    # Generate appropriate model code based on type
    generated_code = _generate_model_code(user_prompt, model_type)

    # Validate generated code syntax
    is_valid, validation_error = _validate_code_syntax(generated_code)

    result = {
        code_key: generated_code,
        model_type_key: model_type,
        "code_valid": is_valid,
        "generation_timestamp": json.dumps({"timestamp": "now"}),
    }

    if not is_valid:
        result["error"] = f"Generated code validation failed: {validation_error}"

    return result


def node_train_model(
    state: Any,
    data_key: str = "data",
    code_key: str = "model_code",
    model_key: str = "trained_model",
    error_key: str = "error",
    training_config_key: str = "training_config",
) -> Dict[str, Any]:
    """
    Execute training code on input data with enhanced error handling and monitoring.
    """
    code = state.get(code_key)
    data = state.get(data_key)
    training_config = state.get(training_config_key, {})

    if not code:
        return {error_key: "No model code provided for training"}

    if not data:
        return {error_key: "No training data provided"}

    try:
        # Execute training with timeout and resource monitoring
        model, training_metrics = _execute_training_with_monitoring(
            code, data, training_config
        )

        return {
            model_key: model,
            "training_metrics": training_metrics,
            "training_successful": True,
            error_key: None,
        }

    except Exception as e:
        return {
            model_key: None,
            error_key: f"Training failed: {str(e)}",
            "training_successful": False,
        }


def node_evaluate_model(
    state: Any,
    model_key: str = "trained_model",
    data_key: str = "data",
    metrics_key: str = "metrics",
    evaluation_config_key: str = "evaluation_config",
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics for the trained model.
    """
    model = state.get(model_key)
    data = state.get(data_key)
    eval_config = state.get(evaluation_config_key, {})

    if not model:
        return {metrics_key: {}, "error": "No trained model available for evaluation"}

    try:
        # Compute comprehensive metrics
        metrics = _compute_comprehensive_metrics(model, data, eval_config)

        # Determine primary metric for optimization decisions
        primary_metric = _determine_primary_metric(metrics, eval_config)

        return {
            metrics_key: metrics,
            "primary_metric": primary_metric,
            "evaluation_successful": True,
        }

    except Exception as e:
        return {
            metrics_key: {},
            "error": f"Evaluation failed: {str(e)}",
            "evaluation_successful": False,
        }


def node_validate_model(
    state: Any,
    model_key: str = "trained_model",
    data_key: str = "data",
    validation_results_key: str = "validation_results",
) -> Dict[str, Any]:
    """
    Perform cross-validation and robustness checks.
    """
    model = state.get(model_key)
    data = state.get(data_key)

    if not model:
        return {
            validation_results_key: {},
            "error": "No model available for validation",
        }

    try:
        # Perform cross-validation
        cv_results = _perform_cross_validation(model, data)

        # Robustness checks
        robustness_results = _check_model_robustness(model, data)

        validation_results = {
            "cross_validation": cv_results,
            "robustness": robustness_results,
            "validation_passed": _assess_validation_results(
                cv_results, robustness_results
            ),
        }

        return {
            validation_results_key: validation_results,
            "validation_successful": True,
        }

    except Exception as e:
        return {
            validation_results_key: {},
            "error": f"Validation failed: {str(e)}",
            "validation_successful": False,
        }


def node_optimize_model(
    state: Any,
    model_key: str = "trained_model",
    data_key: str = "data",
    optimization_results_key: str = "optimization_results",
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization and model tuning.
    """
    model = state.get(model_key)
    data = state.get(data_key)
    current_metrics = state.get("metrics", {})

    if not model:
        return {
            optimization_results_key: {},
            "error": "No model available for optimization",
        }

    try:
        # Perform hyperparameter optimization
        optimized_model, optimization_metrics = _optimize_hyperparameters(model, data)

        # Compare with current performance
        improvement = _calculate_improvement(current_metrics, optimization_metrics)

        optimization_results = {
            "optimized_model": optimized_model,
            "optimization_metrics": optimization_metrics,
            "improvement": improvement,
            "optimization_successful": improvement > 0,
        }

        return {
            optimization_results_key: optimization_results,
            "optimization_improved": improvement > 0,
            model_key: optimized_model if improvement > 0 else model,
        }

    except Exception as e:
        return {
            optimization_results_key: {},
            "error": f"Optimization failed: {str(e)}",
            "optimization_improved": False,
        }


def node_explain_results(
    state: Any,
    metrics_key: str = "metrics",
    result_key: str = "messages",
    role: str = "ModelingAgent",
    title: str = "Model Evaluation Summary",
) -> Dict[str, Any]:
    """
    Generate comprehensive, human-readable summary of modeling results.
    """
    metrics = state.get(metrics_key, {})
    validation_results = state.get("validation_results", {})
    optimization_results = state.get("optimization_results", {})
    error = state.get("error")

    # Build comprehensive summary
    summary = {
        "title": title,
        "timestamp": json.dumps({"timestamp": "now"}),
        "model_type": state.get("model_type", "Unknown"),
        "training_successful": state.get("training_successful", False),
        "evaluation_successful": state.get("evaluation_successful", False),
        "metrics": metrics,
        "validation_results": validation_results,
        "optimization_results": optimization_results,
        "error": error,
        "recommendations": _generate_recommendations(state),
    }

    # Create formatted message
    formatted_content = _format_summary_for_display(summary)
    message = AIMessage(content=formatted_content, role=role)

    return {result_key: [message], "final_summary": summary, "workflow_complete": True}


# -----------------------------------------------------------------------------
# Helper Functions (Placeholders for actual implementations)
# -----------------------------------------------------------------------------


def _detect_model_type(prompt: str) -> str:
    """Detect model type from user prompt"""
    # Implementation: analyze prompt for keywords, return model type
    return "classification"  # placeholder


def _generate_model_code(prompt: str, model_type: str) -> str:
    """Generate model code based on prompt and type"""
    # Implementation: generate appropriate model code
    return "# Generated model code"  # placeholder


def _validate_code_syntax(code: str) -> tuple:
    """Validate generated code syntax"""
    # Implementation: check code syntax
    return True, None  # placeholder


def _execute_training_with_monitoring(code: str, data: Any, config: dict) -> tuple:
    """Execute training with monitoring and timeout"""
    # Implementation: safe code execution with monitoring
    return None, {}  # placeholder


def _compute_comprehensive_metrics(model: Any, data: Any, config: dict) -> dict:
    """Compute comprehensive evaluation metrics"""
    # Implementation: calculate various metrics
    return {"accuracy": 0.85}  # placeholder


def _determine_primary_metric(metrics: dict, config: dict) -> float:
    """Determine primary metric for optimization decisions"""
    # Implementation: select primary metric based on problem type
    return metrics.get("accuracy", 0.0)  # placeholder


def _perform_cross_validation(model: Any, data: Any) -> dict:
    """Perform cross-validation"""
    # Implementation: k-fold cross-validation
    return {"cv_mean": 0.82, "cv_std": 0.05}  # placeholder


def _check_model_robustness(model: Any, data: Any) -> dict:
    """Check model robustness"""
    # Implementation: robustness tests
    return {"robust": True}  # placeholder


def _assess_validation_results(cv_results: dict, robustness_results: dict) -> bool:
    """Assess whether validation passed"""
    # Implementation: validate results assessment
    return True  # placeholder


def _optimize_hyperparameters(model: Any, data: Any) -> tuple:
    """Optimize model hyperparameters"""
    # Implementation: hyperparameter optimization
    return None, {}  # placeholder


def _calculate_improvement(current_metrics: dict, new_metrics: dict) -> float:
    """Calculate improvement between metrics"""
    # Implementation: compare metrics
    return 0.03  # placeholder


def _generate_recommendations(state: dict) -> List[str]:
    """Generate recommendations based on results"""
    # Implementation: generate actionable recommendations
    return ["Consider feature engineering", "Try ensemble methods"]  # placeholder


def _format_summary_for_display(summary: dict) -> str:
    """Format summary for human-readable display"""
    # Implementation: create nicely formatted summary
    return json.dumps(summary, indent=2)  # placeholder
