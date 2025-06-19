#!/usr/bin/env python3
"""
Debug script to test the relocate_imports_inside_function step by step.
"""

import re

from regex import relocate_imports_inside_function


def debug_relocate_function():
    """Debug the relocate_imports_inside_function step by step."""

    # Sample code that simulates what the LLM might generate
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

    print("=== DEBUGGING RELOCATE_IMPORTS_INSIDE_FUNCTION ===")
    print("\n1. Original code:")
    print(sample_code)

    # Test the regex pattern manually
    print("\n2. Testing regex pattern manually:")
    import_pattern = r"^\s*(import\s+[^\n]+|from\s+\S+\s+import\s+[^\n]+)\s*$"
    imports = re.findall(import_pattern, sample_code, re.MULTILINE)
    print(f"Found {len(imports)} imports:")
    for i, imp in enumerate(imports):
        print(f"  {i+1}: {imp}")

    # Test removing imports
    print("\n3. Testing import removal:")
    code_without_imports = re.sub(
        import_pattern, "", sample_code, flags=re.MULTILINE
    ).strip()
    print("Code without imports:")
    print(code_without_imports)

    # Test function pattern
    print("\n4. Testing function pattern:")
    function_pattern = r"(def\s+\w+\s*\(.*?\):)"
    match = re.search(function_pattern, code_without_imports)
    if match:
        print(f"Function found: {match.group(1)}")
        print(f"Function end position: {match.end()}")
    else:
        print("No function found!")

    # Test the full function
    print("\n5. Testing full relocate_imports_inside_function:")
    result = relocate_imports_inside_function(sample_code)
    print("Result:")
    print(result)

    # Check if the result is different from input
    if result != sample_code:
        print("\n✅ SUCCESS: Function modified the code")
    else:
        print("\n❌ FAILURE: Function did not modify the code")


if __name__ == "__main__":
    debug_relocate_function()
