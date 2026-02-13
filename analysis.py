# %%
from pathlib import Path
from src.model import load_data, explore_data, split_data, EDA, data_cleaning, encode_data, select_model, compare_ensembles, tune_hyperparameters, important_features, evaluate_model

# %%
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "term-deposit-marketing-2020.csv"

# %%
df = load_data(DATA_PATH)
numeric_df, cat_df = explore_data(df)

# %%
df = data_cleaning(df, cat_df, numeric_df)

# %%
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target="y", seed=6016)

# %%
EDA(X_train, y_train, X_val, y_val, cat_df, numeric_df)

# %%  
X_train, X_val, y_train, y_val, le_dict, scaler, le_y = encode_data(X_train, X_val, y_train, y_val, cat_df, numeric_df)

# %%
models, predictions = select_model(X_train, X_val, y_train, y_val)
print(models)

# %%
fitted_models, results_df = compare_ensembles(X_train, y_train, X_val, y_val, seed)
print(results_df)

# %%
best_model, best_params, best_score = tune_hyperparameters(X_train, y_train, seed)

# %%
feature_df = important_features(X_train, best_model)
print(feature_df)

# %%
clf_report = evaluate_model(best_model, X_val, y_val)