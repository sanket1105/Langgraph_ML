#!/usr/bin/env python3
"""
Test script to verify that the logging fix works correctly.
This script tests the relocate_imports_inside_function and the H2O agent logging.
"""

import os
import sys

from regex import add_comments_to_top, relocate_imports_inside_function


def test_relocate_imports():
    """Test the relocate_imports_inside_function with sample code."""

    # Sample code that simulates what the LLM might generate (with imports at top level)
    sample_code = """import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional
import optuna
from optuna.samplers import TPESampler
import mlflow
import mlflow.h2o
from contextlib import nullcontext

def test_function(
    enable_optuna: bool = True,
    enable_mlflow: bool = False
):
    # Convert data to DataFrames
    train_df = pd.DataFrame(train_data)
    
    # Initialize H2O
    h2o.init()
    
    return "success"
"""

    print("Original code (with imports at top level):")
    print(sample_code)
    print("\n" + "=" * 50 + "\n")

    # Test the relocation
    relocated_code = relocate_imports_inside_function(sample_code)

    print("Relocated code (imports moved inside function):")
    print(relocated_code)
    print("\n" + "=" * 50 + "\n")

    # Test with comments
    final_code = add_comments_to_top(relocated_code, agent_name="test_agent")

    print("Final code with comments:")
    print(final_code)

    return final_code


def test_logging():
    """Test the logging functionality."""
    from ai_logging import log_ai_function

    # Create test code
    test_code = test_relocate_imports()

    # Test logging
    log_path = "test_logs/"
    file_name = "test_function.py"

    print(f"\nTesting logging to {log_path}{file_name}...")

    file_path, f_name = log_ai_function(
        response=test_code,
        file_name=file_name,
        log=True,
        log_path=log_path,
        overwrite=True,
    )

    print(f"File saved to: {file_path}")
    print(f"File name: {f_name}")

    # Check if file exists and read it
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            content = f.read()
        print(f"\nFile content length: {len(content)} characters")
        print("File created successfully!")

        # Show the first few lines to verify structure
        lines = content.split("\n")
        print("\nFirst 20 lines of the file:")
        for i, line in enumerate(lines[:20]):
            print(f"{i+1:2d}: {line}")
    else:
        print("ERROR: File was not created!")

    return file_path


if __name__ == "__main__":
    print("Testing logging fix...")
    print("=" * 50)

    # Test import relocation
    test_relocate_imports()

    print("\n" + "=" * 50)
    print("Testing logging functionality...")

    # Test logging
    test_logging()

    print("\n" + "=" * 50)
    print("Test completed!")
