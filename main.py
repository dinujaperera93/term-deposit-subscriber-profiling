import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import random
import duckdb
from src.two_layer_model import load_data, explore_data, train_two_layer_pipeline
from src.cluster_model import cluster_subscribers

def main():
    seed = random.randint(1000, 9999)
    print("Random seed: ", seed)

    ROOT = Path(__file__).resolve().parent
    DATA_PATH = ROOT / "data" / "term-deposit-marketing-2020.csv"

    term_deposit_df = load_data(DATA_PATH)
    numeric_df, categorical_df = explore_data(term_deposit_df)

    results = train_two_layer_pipeline(term_deposit_df, seed, categorical_df, numeric_df)

    # Select subscribers from term_deposit_df
    subscribers = duckdb.sql("""
        SELECT *
        FROM term_deposit_df
        WHERE y = 'yes'
    """).df()

    print(f"Subscribers: {len(subscribers):,}")
    cluster_subscribers(subscribers, seed=seed, save_dir='figures_Clustering')

if __name__ == "__main__":
    main()