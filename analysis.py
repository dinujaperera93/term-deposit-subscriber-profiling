# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
# %matplotlib agg

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import random

from src.model import (
    load_data, explore_data, split_data, EDA, data_cleaning,
    encode_data, select_model, compare_ensembles,
    tune_hyperparameters, important_features, evaluate_model
)

# %%
seed = random.randint(1000, 9999)
print("Random seed: ", seed)

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "term-deposit-marketing-2020.csv"

# %%
term_deposit_df = load_data(DATA_PATH)
numeric_df, categorical_df = explore_data(term_deposit_df)

# %%
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    term_deposit_df, target="y", seed=seed
)

# %%
df_tv = EDA(X_train, y_train, X_val, y_val, categorical_df, numeric_df)

# %%
term_deposit_df, cat_mode, num_bounds = data_cleaning(
    df_tv, categorical_df, numeric_df
)

# %%
X_train, X_val, y_train, y_val, le_dict, scaler, le_y = encode_data(
    X_train, X_val, y_train, y_val, categorical_df, numeric_df
)

# %%
models, predictions = select_model(X_train, X_val, y_train, y_val)
print(models)

# %%
fitted_models, results_df = compare_ensembles(
    X_train, y_train, X_val, y_val, seed, cv=5
)
print(results_df)

# %%
best_model, best_params, best_score = tune_hyperparameters(
    X_train, y_train, X_val, y_val, seed
)

# %%
feature_df = important_features(X_train, best_model)
print(feature_df)

# %%
clf_report = evaluate_model(
    best_model, X_test, y_test,
    categorical_df, numeric_df,
    cat_mode, num_bounds, le_dict, scaler, le_y
)
print(clf_report)
