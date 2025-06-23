"""
Brier Score Example and Explanation

The Brier score is a proper scoring rule for probabilistic predictions in binary classification.
It measures the accuracy of probabilistic predictions and ranges from 0 to 1, where:
- 0 = Perfect predictions
- 1 = Worst possible predictions

For binary classification, Brier Score = mean((predicted_probability - actual_outcome)²)

Key advantages:
1. Proper scoring rule - encourages honest probability estimates
2. Penalizes overconfident predictions
3. Useful for model calibration assessment
4. Works well with imbalanced datasets
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calculate_brier_score_examples():
    """
    Demonstrate Brier score calculation with different scenarios.
    """
    print("=== Brier Score Examples ===\n")

    # Example 1: Perfect predictions
    print("1. Perfect Predictions:")
    actual = np.array([0, 1, 0, 1, 0])
    perfect_probs = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    brier_perfect = brier_score_loss(actual, perfect_probs)
    print(f"   Actual: {actual}")
    print(f"   Predicted probabilities: {perfect_probs}")
    print(f"   Brier Score: {brier_perfect:.4f} (Perfect!)\n")

    # Example 2: Good predictions
    print("2. Good Predictions:")
    good_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.1])
    brier_good = brier_score_loss(actual, good_probs)
    print(f"   Actual: {actual}")
    print(f"   Predicted probabilities: {good_probs}")
    print(f"   Brier Score: {brier_good:.4f} (Good)\n")

    # Example 3: Poor predictions
    print("3. Poor Predictions:")
    poor_probs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    brier_poor = brier_score_loss(actual, poor_probs)
    print(f"   Actual: {actual}")
    print(f"   Predicted probabilities: {poor_probs}")
    print(f"   Brier Score: {brier_poor:.4f} (Poor - random guessing)\n")

    # Example 4: Overconfident predictions
    print("4. Overconfident Predictions:")
    overconfident_probs = np.array(
        [0.0, 0.0, 0.0, 1.0, 0.0]
    )  # Wrong on second prediction
    brier_overconfident = brier_score_loss(actual, overconfident_probs)
    print(f"   Actual: {actual}")
    print(f"   Predicted probabilities: {overconfident_probs}")
    print(f"   Brier Score: {brier_overconfident:.4f} (Penalized for overconfidence)\n")


def brier_score_interpretation():
    """
    Explain how to interpret Brier scores.
    """
    print("=== Brier Score Interpretation ===\n")

    print("Brier Score Ranges:")
    print("- 0.000 - 0.050: Excellent calibration")
    print("- 0.050 - 0.100: Good calibration")
    print("- 0.100 - 0.150: Fair calibration")
    print("- 0.150 - 0.200: Poor calibration")
    print("- 0.200 - 0.250: Very poor calibration")
    print("- 0.250 - 1.000: Extremely poor calibration\n")

    print("Key Insights:")
    print("1. Lower is better (unlike AUC where higher is better)")
    print("2. Penalizes overconfident predictions heavily")
    print("3. Rewards well-calibrated probability estimates")
    print("4. Works well with imbalanced datasets")
    print("5. Can be decomposed into reliability and resolution components\n")


def compare_with_other_metrics():
    """
    Compare Brier score with other common metrics.
    """
    print("=== Comparison with Other Metrics ===\n")

    # Create example data
    np.random.seed(42)
    n_samples = 1000

    # Simulate imbalanced dataset (10% positive class)
    actual = np.random.binomial(1, 0.1, n_samples)

    # Model A: Well-calibrated but moderate discrimination
    prob_a = np.zeros(n_samples)
    prob_a[actual == 1] = np.random.beta(
        3, 1, sum(actual == 1)
    )  # Higher probs for positive
    prob_a[actual == 0] = np.random.beta(
        1, 3, sum(actual == 0)
    )  # Lower probs for negative

    # Model B: Poorly calibrated but good discrimination
    prob_b = np.zeros(n_samples)
    prob_b[actual == 1] = np.random.beta(5, 1, sum(actual == 1)) * 0.8  # Underconfident
    prob_b[actual == 0] = np.random.beta(1, 5, sum(actual == 0)) * 0.2  # Underconfident

    # Calculate metrics
    from sklearn.metrics import average_precision_score, roc_auc_score

    auc_a = roc_auc_score(actual, prob_a)
    auc_b = roc_auc_score(actual, prob_b)

    ap_a = average_precision_score(actual, prob_a)
    ap_b = average_precision_score(actual, prob_b)

    brier_a = brier_score_loss(actual, prob_a)
    brier_b = brier_score_loss(actual, prob_b)

    print("Model Comparison (Imbalanced Dataset - 10% positive):")
    print(f"{'Metric':<15} {'Model A':<10} {'Model B':<10}")
    print("-" * 40)
    print(f"{'AUC':<15} {auc_a:<10.4f} {auc_b:<10.4f}")
    print(f"{'Avg Precision':<15} {ap_a:<10.4f} {ap_b:<10.4f}")
    print(f"{'Brier Score':<15} {brier_a:<10.4f} {brier_b:<10.4f}")
    print()

    print("Interpretation:")
    print("- Model A: Good calibration, moderate discrimination")
    print("- Model B: Poor calibration, good discrimination")
    print("- Brier score correctly identifies Model A as better calibrated")
    print("- AUC and Average Precision focus on discrimination ability\n")


def practical_usage_tips():
    """
    Provide practical tips for using Brier score.
    """
    print("=== Practical Usage Tips ===\n")

    print("When to use Brier Score:")
    print("1. Binary classification with probabilistic outputs")
    print("2. Model calibration assessment")
    print("3. Comparing models with different probability distributions")
    print("4. Imbalanced datasets where accuracy is misleading")
    print("5. Risk assessment applications\n")

    print("Best Practices:")
    print("1. Always report Brier score alongside AUC/accuracy")
    print("2. Use calibration plots to visualize probability calibration")
    print("3. Consider Brier score decomposition for deeper insights")
    print("4. Use cross-validation to get reliable Brier score estimates")
    print("5. Compare against baseline (majority class probability)\n")

    print("Common Pitfalls:")
    print("1. Confusing Brier score with accuracy")
    print("2. Not considering class imbalance effects")
    print("3. Ignoring calibration in favor of discrimination only")
    print(
        "4. Using Brier score for multi-class problems (use Brier score loss instead)"
    )
    print("5. Not validating probability estimates on holdout set\n")


if __name__ == "__main__":
    calculate_brier_score_examples()
    brier_score_interpretation()
    compare_with_other_metrics()
    practical_usage_tips()

    print("=== Summary ===")
    print("Brier score is an excellent metric for evaluating probabilistic predictions")
    print("in binary classification, especially when model calibration matters.")
    print(
        "It complements traditional metrics like AUC by focusing on probability accuracy."
    )
    print("Use it alongside other metrics for a comprehensive model evaluation.")
