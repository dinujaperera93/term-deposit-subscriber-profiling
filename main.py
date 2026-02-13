import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import random
from src.model import (load_data, explore_data, split_data, EDA, data_cleaning, 
                       encode_data, select_model, compare_ensembles, 
                       tune_hyperparameters, important_features, evaluate_model)

def main():
    seed = random.randint(1000, 9999)
    print("Random seed: ", seed)

    ROOT = Path(__file__).resolve().parent
    DATA_PATH = ROOT / "data" / "term-deposit-marketing-2020.csv"
    
    term_deposit_df = load_data(DATA_PATH)
    numeric_cols, categorical_cols = explore_data(term_deposit_df)
    
    term_deposit_df = data_cleaning(term_deposit_df, categorical_cols, numeric_cols)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(term_deposit_df, target="y", seed=seed)
    EDA(X_train, y_train, X_val, y_val, categorical_cols, numeric_cols)
    
    X_train, X_val, y_train, y_val, le_dict, scaler, le_y= encode_data(X_train, X_val, y_train, y_val, categorical_cols, numeric_cols)
    
    models, predictions = select_model(X_train, X_val, y_train, y_val)
    print(models)
        
    fitted_models, results_df = compare_ensembles(X_train, y_train, X_val, y_val, seed)
    print(results_df)

    best_model, best_params, best_score = tune_hyperparameters(X_train, y_train, seed)

    feature_df = important_features(X_train, best_model)
    print(feature_df)

    clf_report = evaluate_model(best_model, X_val, y_val)

if __name__ == "__main__":
    main()