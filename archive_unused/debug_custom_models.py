#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from h20 import H2OMLAgentEnhanced

# 1. Create sample data
X, y = make_classification(
    n_samples=100, n_features=10, n_informative=5, n_redundant=2, random_state=42
)
X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y_series = pd.Series(y, name="target")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42
)

# 3. Train custom models
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
lr = LogisticRegression(solver="liblinear")
lr.fit(X_train, y_train)

custom_models = {
    "Random Forest": rf,
    "Logistic Regression (L2)": lr,
}

# 4. Create a fake leaderboard with both H2O and custom model IDs
leaderboard_df = pd.DataFrame(
    {
        "model_id": [
            "GBM_1_AutoML_1_20230601_123456",
            "Random Forest",
            "Logistic Regression (L2)",
        ]
    }
)

# 5. Create a test H2OFrame
import h2o

h2o.init()
test_h2o = h2o.H2OFrame(X_test)
test_h2o["target"] = h2o.H2OFrame(y_test.to_frame())

# 6. Call the combined plotting function
agent = H2OMLAgentEnhanced(model=None)
pdf_path = agent.plot_combined_reliability_curves(
    leaderboard_df=leaderboard_df,
    test_h2o=test_h2o,
    target_variable="target",
    custom_models=custom_models,
    logs_dir="logs/",
)
print(f"Combined plot saved to: {pdf_path}")

# 7. Call the custom models only plotting function
custom_pdf_path = agent.plot_custom_models_reliability_curves(
    custom_models=custom_models,
    test_h2o=test_h2o,
    target_variable="target",
    logs_dir="logs/",
)
print(f"Custom models only plot saved to: {custom_pdf_path}")

h2o.cluster().shutdown()
