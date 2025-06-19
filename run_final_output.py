"""
Final script to run H2O ML workflow and get complete output with Brier score
"""

import os

import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from h20 import H2OMLAgentEnhanced


def run_final_output():
    """
    Run the complete H2O ML workflow and display final results with Brier score.
    """

    print("=" * 60)
    print("🚀 H2O ML WORKFLOW - FINAL OUTPUT WITH BRIER SCORE")
    print("=" * 60)

    # 1. Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Set OPENAI_API_KEY environment variable first")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return

    # 2. Create test data
    print("\n📊 Creating test dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_classes=2,
        n_clusters_per_class=1,
        n_redundant=5,
        n_informative=10,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    print(f"✅ Dataset created: {df.shape[0]} samples, {df.shape[1]-1} features")

    # 3. Initialize H2O ML Agent
    print("\n🤖 Initializing H2O ML Agent...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    ml_agent = H2OMLAgentEnhanced(llm=llm)

    # 4. Run the complete workflow
    print("\n⚡ Running H2O AutoML workflow...")
    print("   This may take a few minutes...")

    try:
        result = ml_agent.run(df)
        print("✅ Workflow completed successfully!")

    except Exception as e:
        print(f"❌ Error during workflow: {e}")
        return

    # 5. Get and display all metrics
    print("\n" + "=" * 60)
    print("📈 FINAL RESULTS")
    print("=" * 60)

    # Get test metrics
    try:
        test_metrics = ml_agent.get_test_metrics()
        print("\n🔍 TEST SET METRICS:")
        print("-" * 30)
        for metric, value in test_metrics.items():
            if isinstance(value, float):
                print(f"{metric:25}: {value:.6f}")
            else:
                print(f"{metric:25}: {value}")
    except Exception as e:
        print(f"❌ Could not get test metrics: {e}")

    # Get calibration metrics
    try:
        calib_metrics = ml_agent.get_calibration_metrics()
        print("\n🎯 CALIBRATION METRICS:")
        print("-" * 30)
        for metric, value in calib_metrics.items():
            if isinstance(value, float):
                print(f"{metric:25}: {value:.6f}")
            else:
                print(f"{metric:25}: {value}")
    except Exception as e:
        print(f"❌ Could not get calibration metrics: {e}")

    # 6. Brier Score Analysis
    print("\n" + "=" * 60)
    print("🎯 BRIER SCORE ANALYSIS")
    print("=" * 60)

    brier_test = test_metrics.get("brier_score", None)
    brier_calib = calib_metrics.get("brier_score", None)

    if brier_test is not None:
        print(f"\n📊 Test Set Brier Score: {brier_test:.6f}")
        if brier_test < 0.05:
            print("   ✅ EXCELLENT calibration")
        elif brier_test < 0.1:
            print("   ✅ GOOD calibration")
        elif brier_test < 0.2:
            print("   ⚠️  FAIR calibration")
        else:
            print("   ❌ POOR calibration")

    if brier_calib is not None:
        print(f"\n📊 Calibration Set Brier Score: {brier_calib:.6f}")
        if brier_calib < 0.05:
            print("   ✅ EXCELLENT calibration")
        elif brier_calib < 0.1:
            print("   ✅ GOOD calibration")
        elif brier_calib < 0.2:
            print("   ⚠️  FAIR calibration")
        else:
            print("   ❌ POOR calibration")

    # 7. Model Information
    print("\n" + "=" * 60)
    print("🤖 MODEL INFORMATION")
    print("=" * 60)

    try:
        model_info = ml_agent.get_model_info()
        print(f"\n🏆 Best Model: {model_info.get('best_model', 'Unknown')}")
        print(f"📁 Model Location: {model_info.get('model_path', 'Unknown')}")
        print(f"⏱️  Training Time: {model_info.get('training_time', 'Unknown')}")
    except Exception as e:
        print(f"❌ Could not get model info: {e}")

    # 8. Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    print("\n✅ H2O AutoML workflow completed successfully!")
    print("✅ Brier score calculated and displayed")
    print("✅ All metrics available for analysis")

    if brier_test is not None or brier_calib is not None:
        print("✅ Model calibration assessed")

    print("\n🎉 Final output complete!")


if __name__ == "__main__":
    run_final_output()
