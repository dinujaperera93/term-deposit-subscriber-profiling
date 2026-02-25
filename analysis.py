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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Term Deposit Subscription Prediction
# **European Banking | Customer Intention Classification | Telemarketing Optimisation**
#
# ---

# %% [markdown]
# ## Business Context
#
# A European banking institution runs direct marketing campaigns via phone calls to
# promote term deposit subscriptions. The current approach contacts customers
# indiscriminately, resulting in low conversion rates, wasted agent resources, and
# a poor customer experience.
#
# **Goals:**
# 1. Predict whether a customer will subscribe to a term deposit (`y`)
# 2. Determine which features most strongly drive subscription decisions
#
# **Success Metric:** 81%+ accuracy via 5-fold cross-validation (average score)
# **Key Operational Metric:** Minority-class recall — catching actual subscribers

# %% [markdown]
# ## Solution Design: Two-Layer Pipeline
#
# Features fall into two groups separated by a temporal boundary:
#
# | Layer | Features Available | Purpose |
# |-------|--------------------|---------|
# | **Model 1 : Pre-call** | Demographics + financial history | Predict who to call *before* any contact |
# | **Model 2 : Post-call** | All features (incl. call data) | Predict who to *continue* calling after contact |

# %% [markdown]
# ---
# ## Import Dependencies

# %%
# %matplotlib inline

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import random
import duckdb

from src.cluster_model import cluster_subscribers
from src.two_layer_model import (
    load_data, get_feature_sets, explore_data,
    split_data, data_cleaning, encode_data,
    select_model, compare_ensembles, tune_hyperparameters,
    feature_importance, evaluate_model
)

SEED = random.randint(1000, 9999)
print(f"Seed: {SEED}")

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "term-deposit-marketing-2020.csv"

# %% [markdown]
# ---
# ## 1. Data Loading

# %%
term_deposit_df = load_data(DATA_PATH)
print(f"Shape: {term_deposit_df.shape[0]:,} rows x {term_deposit_df.shape[1]} columns")
term_deposit_df.head()

# %% [markdown]
# ---
# ## 2. Feature Groups
#
# Features are split into pre-call and post-call groups based on when they become available.
# `duration` is in the post-call group — it only exists after a call ends and must not
# be used in Model 1 (data leakage).

# %%
pre_call_cols, post_call_cols = get_feature_sets(term_deposit_df)
all_cols = pre_call_cols + post_call_cols
print(f"Model 1 — Pre-call  : {pre_call_cols}")
print(f"Model 2 adds        : {post_call_cols}")

# %% [markdown]
# ---
# ## 3. Exploratory Data Analysis
#
# EDA covers target distribution, feature distributions by class, numeric histograms,
# correlation structure, and identification of structural missing values (`"unknown"`).

# %%
numeric_df, categorical_df = explore_data(term_deposit_df)

# %% [markdown]
# ### Key Findings
#
# #### Class Imbalance
#
# | Class | Count | Share |
# |-------|-------|-------|
# | No (did not subscribe) | 37104 | 92.8% |
# | Yes (subscribed) | 2896 | 7.2% |
#
# The dataset is highly imbalanced. A majority-class classifier achieves 93% accuracy
# while being useless to the business. This drives two decisions:
# - **Metric:** Optimise for minority-class recall alongside the 81% accuracy target
# - **Model config:** `class_weight='balanced'` to penalise missed subscribers
#
# ---
#
# #### Numeric Variables
#
# - **`age`:** Range 18–95, mean ~41. No anomalies.
# - **`balance`:** Average yearly balance in euros. Can be negative (overdrafts).
#   Range −8,019 to 102,127; highly right-skewed with significant outliers.
# - **`day`:** Day of month of last contact (1–31). Not a duration. Most calls have been made around 20th
# - **`duration`:** Last call duration in seconds. Highly predictive but only known
#   *after* the call ends; excluded from Model 1 to prevent data leakage.
# - **`campaign`:** Number of contacts made this campaign. Most customers are contacted
#   1–3 times; the distribution has a heavy right tail. Maximum value is 63 which seems an outlier.
#
# ---
#
# #### Categorical Variables
#
# - **`job`:** Management, blue-collar, and technician are the three most common categories.
# - **`marital`:** Married customers are the majority.
# - **`education`:** Secondary education is the most frequent level.
# - **`default`:** Very few customers have credit in default — a rare event in the dataset.
# - **`housing`:** Roughly evenly split between customers with and without a housing loan.
# - **`loan`:** Most customers do not hold a personal loan.
# - **`contact`:** The majority of contacts were made via cellular. A substantial portion
#   of records have an unknown contact type (see table below).
# - **`month`:** May has the highest contact volume by a large margin. No calls were made
#   in September, suggesting campaign scheduling constraints.
#
# ---
#
# #### Structural Missing Values ("unknown")
#
# Pandas reports zero nulls but it is assumed that the missingness is encoded as the string `"unknown"`:
#
# | Column | Count | % of Dataset |
# |--------|-------|-------------|
# | `contact` | 12,765 | 31.9% |
# | `education` | 1,531 | 3.8% |
# | `job` | 235 | 0.6% |
#
# **Overlap analysis** — rows where multiple columns are unknown simultaneously:
#
# | Pair | Rows |
# |------|------|
# | `contact` & `education` | 666 |
# | `contact` & `job` | 110 |
# | `education` & `job` | 104 |
# | All three | 54 |
# | Any one (union) | 13,705 (34.3%) |
#
# **Why rows were not dropped:**
# Dropping any row containing "unknown" would remove 34.3% of the dataset, nearly all of it
# driven by `contact` alone. Given the existing class imbalance, discarding that volume
# would significantly reduce minority-class representation in training.
#
# **Design decision — hybrid imputation strategy:**
# - **`contact`** → "unknown" retained as a valid category. At 31.9%, this is not random
#   missingness but a distinct cohort (customers reached via an unlogged channel). LabelEncoder assigns
#   it its own integer; LightGBM can learn from it.
# - **`education` and `job`** → mode-imputed. Sparse unknowns (3.8% and 0.6%) with no
#   evidence they form a meaningful segment. Mode imputation is simple and introduces
#   negligible bias at this volume.
#
# ---
#
# #### Outliers
#
# IQR-based clipping `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]` is applied to all numeric features.
# Bounds are computed on training data only to prevent leakage.

# %% [markdown]
# ---
# ## 4. Preprocessing Pipeline
#
# All transformations are **fitted on training data only**; no leakage at any stage.
#
# | Step | Method | Rationale |
# |------|--------|-----------|
# | Train / Val / Test split | Stratified 80 / 10 / 10 | Preserves the 93/7 class ratio across all sets |
# | `contact` "unknown" | Kept as valid category | 31.9% — informative cohort, not random missingness |
# | `education`, `job` "unknown" | Mode imputation | Sparse unknowns; mode fitted on train only |
# | Outlier handling | IQR clipping | Reduces outlier influence without removing rows |
# | Categorical encoding | `LabelEncoder` | Converts string categories to integers; compatible with all sklearn estimators |
# | Numeric scaling | `StandardScaler` | Features span mixed distributions (`balance`, `duration`, `campaign` are skewed; `age` near-normal; `day` uniform). After IQR clipping, extreme outliers are removed; StandardScaler centres and scales each feature to zero mean and unit variance without distortion |
# | Target encoding | `LabelEncoder` | `no` → 0, `yes` → 1 |

# %% [markdown]
# ### Step 4.1 — Train / Validation / Test Split

# %%
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    term_deposit_df, target="y", seed=SEED
)

# %% [markdown]
# ### Step 4.2 — Data Cleaning
#
# Outlier clipping (IQR) fitted on train, applied to val.
# Mode imputation for `education` and `job` only; `contact` "unknown" kept as-is.

# %%
X_train_cleaned, X_val_cleaned, cat_mode, num_bounds, cat_cols, num_cols = data_cleaning(
    X_train, X_val, categorical_df, numeric_df
)
print("Mode values used for imputation:", cat_mode)

# %% [markdown]
# ---
# ## 5. Model Selection
#
# **LazyPredict** benchmarks 30+ classifiers in a single pass, ranked by minority-class
# recall on the validation set.
#
# **Selected: LightGBM (`LGBMClassifier`)**
#
# | Criterion | LightGBM |
# |-----------|----------|
# | Minority-class recall | Consistently top-ranked |
# | Class imbalance | Native `class_weight='balanced'` support |
# | Interpretability | Feature importance for client-facing insights |
# | Speed | Practical for repeated tuning and production retraining |

# %% [markdown]
# ---
# ## 6. Hyperparameter Tuning
#
# **Hyperopt** with TPE (Tree-structured Parzen Estimator) performs Bayesian optimisation.
# Objective: maximise average minority-class recall across 5-fold cross-validation.
#
# | Parameter | Range | Purpose |
# |-----------|-------|---------|
# | `n_estimators` | 50–500 | Boosting iterations |
# | `max_depth` | 2–8 | Tree depth, controls overfitting |
# | `learning_rate` | 0.05–0.2 | Step size shrinkage |
# | `num_leaves` | 5–31 | Tree complexity |
# | `min_child_samples` | 10–30 | Leaf regularisation |
# | `subsample` | 0.7–0.9 | Row sampling |
# | `colsample_bytree` | 0.7–0.9 | Feature sampling |
#
# TPE converges in ~50 evaluations by learning from prior trials, where grid search
# across this space would require ~1,000+.

# %% [markdown]
# ---
# ## 7. Model 1 — Pre-Call Targeting
# *Feature set: demographics + financial history only (no call data)*

# %% [markdown]
# ### Step 7.1 — Encode Features (Model 1)

# %%
pre_cat_cols = [c for c in pre_call_cols if c in cat_cols]
pre_num_cols = [c for c in pre_call_cols if c in num_cols]

X_tr_m1, X_va_m1, y_tr_enc, y_va_enc, le1, sc1, le_y1 = encode_data(
    X_train_cleaned[pre_call_cols], X_val_cleaned[pre_call_cols],
    y_train, y_val, pre_cat_cols, pre_num_cols
)
print(f"Model 1 training shape : {X_tr_m1.shape}")
print(f"Features               : {X_tr_m1.columns.tolist()}")

# %% [markdown]
# ### Step 7.2 — Model Selection (Model 1)

# %%
models_df1, _ = select_model(X_tr_m1, X_va_m1, y_tr_enc, y_va_enc)
print(models_df1)

# %% [markdown]
# ### Step 7.3 — Ensemble Comparison (Model 1)
#
# Compares tree-based, distance-based, and ensemble strategies via 5-fold CV minority recall.
# The best-performing architecture is then carried forward to hyperparameter tuning.

# %%
fitted_models1, ensemble_df1 = compare_ensembles(X_tr_m1, y_tr_enc, SEED)
print(ensemble_df1)

# %% [markdown]
# ### Step 7.4 — Hyperparameter Tuning (Model 1)

# %%
model1, params1, score1 = tune_hyperparameters(X_tr_m1, y_tr_enc, X_va_m1, y_va_enc, SEED)

# %% [markdown]
# ### Step 7.5 — Feature Importance (Model 1)

# %%
feat1 = feature_importance(X_tr_m1, model1)
print(feat1)

# %% [markdown]
# ### Step 7.6 — Evaluation on Test Set (Model 1)

# %%
report1, cm1 = evaluate_model(
    model1, X_test[pre_call_cols].copy(), y_test,
    le1, sc1, le_y1,
    pre_cat_cols, pre_num_cols, cat_mode, num_bounds, pre_call_cols, "Model1"
)
print(report1)

# %% [markdown]
# ---
# ## 8. Model 2 — Post-Call Follow-Up
# *Feature set: all features including call data (`duration`, `contact`, `month`, `day`, `campaign`)*

# %% [markdown]
# ### Step 8.1 — Encode Features (Model 2)

# %%
all_cat_cols = [c for c in all_cols if c in cat_cols]
all_num_cols = [c for c in all_cols if c in num_cols]

X_tr_m2, X_va_m2, y_tr_enc2, y_va_enc2, le2, sc2, le_y2 = encode_data(
    X_train_cleaned[all_cols], X_val_cleaned[all_cols],
    y_train, y_val, all_cat_cols, all_num_cols
)
print(f"Model 2 training shape : {X_tr_m2.shape}")
print(f"Features               : {X_tr_m2.columns.tolist()}")

# %% [markdown]
# ### Step 8.2 — Model Selection (Model 2)

# %%
models_df2, _ = select_model(X_tr_m2, X_va_m2, y_tr_enc2, y_va_enc2)
print(models_df2)

# %% [markdown]
# ### Step 8.3 — Ensemble Comparison (Model 2)

# %%
fitted_models2, ensemble_df2 = compare_ensembles(X_tr_m2, y_tr_enc2, SEED)
print(ensemble_df2)

# %% [markdown]
# ### Step 8.4 — Hyperparameter Tuning (Model 2)

# %%
model2, params2, score2 = tune_hyperparameters(X_tr_m2, y_tr_enc2, X_va_m2, y_va_enc2, SEED)

# %% [markdown]
# ### Step 8.5 — Feature Importance (Model 2)

# %%
feat2 = feature_importance(X_tr_m2, model2)
print(feat2)

# %% [markdown]
# ### Step 8.6 — Evaluation on Test Set (Model 2)

# %%
report2, cm2 = evaluate_model(
    model2, X_test[all_cols].copy(), y_test,
    le2, sc2, le_y2,
    all_cat_cols, all_num_cols, cat_mode, num_bounds, all_cols, "Model2"
)
print(report2)

# %% [markdown]
# ---
# ## 9. Results Summary

# %%
print("Model 1: Who to Call (Pre-Call Targeting)")
print("  Business question : Which customers are worth contacting?")
print(f"  CV Minority Recall : {score1:.4f}")
print("  Feature Importance:")
print(feat1.to_string(index=False))

print("\nModel 2: Who Will Subscribe (Post-Call Follow-Up)")
print("  Business question : Of those contacted, who is likely to subscribe?")
print(f"  CV Minority Recall : {score2:.4f}")
print("  Feature Importance:")
print(feat2.to_string(index=False))

# %% [markdown]
# ### Business Impact

# %%
CAMPAIGN_SIZE = 40_000

# Model 1 confusion matrix: rows = actual, cols = predicted
# [TN, FP]   --> actual Do Not Call
# [FN, TP]   --> actual Call (subscriber)
tn1, fp1, fn1, tp1 = cm1.ravel()
total1 = tn1 + fp1 + fn1 + tp1

predicted_no_call   = tn1 + fn1          # model says "Do Not Call"
predicted_call      = tp1 + fp1          # model says "Call"
pct_saved           = predicted_no_call / total1

calls_saved         = int(pct_saved * CAMPAIGN_SIZE)
useless_calls_test  = fp1                # called but won't subscribe
missed_subs_test    = fn1               # missed subscribers
correct_calls_test  = tp1              # correctly targeted

avg_duration_sec     = X_train['duration'].mean()           # avg call duration from train only
avg_duration_min     = avg_duration_sec / 60
hours_saved_test     = predicted_no_call * avg_duration_sec / 3600
hours_saved_campaign = calls_saved * avg_duration_sec / 3600

print(f"Business Impact: Model 1 (Pre-Call Targeting)")
print(f"  Test set size               : {total1:,}")
print(f"  Predicted 'Do Not Call'     : {predicted_no_call:,}  ({100 * pct_saved:.1f}% of test set)")
print(f"  Predicted 'Call'            : {predicted_call:,}")
print()
print(f"  Scaled to {CAMPAIGN_SIZE:,} campaign contacts:")
print(f"    Calls avoided (saved manpower) by the company is : {calls_saved:,}")
print()
print(f"  Average call duration       : {avg_duration_sec:.0f}s  ({avg_duration_min:.1f} min)")
print(f"  Hours saved (test)    : {hours_saved_test:.1f} hours")
print(f"  Hours saved ({CAMPAIGN_SIZE:,})  : {hours_saved_campaign:.0f} hours  ({hours_saved_campaign/8:.0f} working days)")
print()
print(f"  Within the {predicted_call:,} calls made (test set):")
print(f"    Correctly targeted        : {correct_calls_test:,}  (likely subscribers)")
print(f"    Useless calls             : {useless_calls_test:,}  (won't subscribe)")
print(f"    Missed subscribers        : {missed_subs_test:,}  (false negatives)")

# %% [markdown]
# ---
# ## 10. Subscriber Segmentation — KMeans Clustering

# %%
subscribers = duckdb.sql("""
    SELECT *
    FROM term_deposit_df
    WHERE y = 'yes'
""").df()

print(f"Subscribers: {len(subscribers):,}")
cluster_subscribers(subscribers)


# %% [markdown]
# ### Plot 1 — Combined 2D Pairs and 3D Triplets (Standardised)
#
# **What it shows:** A 4×3 figure — top 2 rows contain 6 pairwise 2D scatter plots across all numerical feature combinations; bottom 2 rows contain 4 three-dimensional scatter plots for the most meaningful feature triplets (age/balance/duration, balance/duration/campaign, age/duration/campaign, campaign/day/duration).
#
# **Conclusion:** No tight clusters are visible in any 2D or 3D view before clustering is applied. Subscribers are spread heterogeneously across all feature combinations, confirming that no single pair or triplet alone separates them. This motivates running KMeans across all 5 features simultaneously to detect hidden structure.
#
# ---
#
# ### Plot 2 — KMeans k=2 to k=19 (3×6 grid, day vs campaign)
#
# **What it shows:** How the day–campaign space is partitioned as k increases from 2 to 19. Clustering runs on all 5 features; day and campaign are used as visualisation axes. Yellow markers are centroids projected onto these two axes.
#
# **Conclusion:** At low k (2–3), clusters separate subscribers along the campaign axis — those contacted fewer times vs those requiring persistent follow-up. Beyond k=4–5, clusters fragment without forming meaningful new segments. The day axis shows weaker separation, consistent with its near-zero feature importance. The silhouette score and elbow method confirm the optimal k.
#
# ---
#
# ### Plot 3 — Silhouette Scores (3×6 grid, k=2 to k=19, day vs campaign)
#
# **What it shows:** Each subplot shows cluster assignments for a given k alongside its silhouette score on day vs campaign axes. Silhouette ranges from −1 to 1; higher = denser, better-separated clusters.
#
# **Conclusion:** The silhouette score peaks at **k=2**, confirming two clusters as the most statistically justified segmentation. Beyond k=4 scores plateau or decline, indicating additional clusters capture noise rather than structure.
#
# ---
#
# ### Plot 4 — Cluster Describe (k=2)
#
# **What it shows:** Descriptive statistics (`mean`, `std`, `min`, `max`) for each of the two clusters across all 5 standardised features.
#
# **Conclusion — Two subscriber segments emerge:**
#
# | | Cluster 0 | Cluster 1 |
# |---|---|---|
# | `age` | Lower (younger) | Higher (older) |
# | `balance` | Lower | Higher |
# | `duration` | Longer calls | Shorter calls |
# | `campaign` | More contacts needed | Fewer contacts needed |
# | `day` | Later in month | Earlier in month |
#
# - **Cluster 0 — High-effort converters:** Younger, lower-balance subscribers who required more campaign contacts and longer calls. Likely first-time savers who needed more persuasion. Higher cost per conversion.
# - **Cluster 1 — Low-effort converters:** Older, higher-balance subscribers who converted with fewer contacts and shorter calls. Experienced savers with existing financial products. Lower cost per conversion and higher lifetime value.
#
# ---
#
# ### Plot 5 — Correlated Feature Pairs (2×3 grid, 6 panels)
#
# **Row 1**
#
# **age vs balance (corr: 0.08)**
# Clusters separate primarily along the balance axis. Financial position — not age alone — drives which segment a subscriber falls into.
#
# **age vs duration (corr: −0.04)**
# Younger subscribers tend to have longer call durations. Agents should budget more time when targeting younger segments.
#
# **day vs campaign (corr: 0.17)**
# The strongest correlation. Subscribers contacted later in the month required more campaign attempts — earlier-month contacts are higher-quality leads.
#
# **Row 2**
#
# **age vs campaign (corr: 0.02)**
# Negligible correlation. Campaign contact count is not tied to age — both young and old subscribers can require multiple contacts.
#
# **day vs duration (corr: −0.03)**
# Negligible correlation. Call duration is unaffected by which day of the month the contact was made.
#
# **balance vs duration (corr: 0.17)**
# Subscribers with higher balances tend to have slightly longer calls, suggesting more thorough conversations with financially engaged customers.
#
# ---
#
# ### Plot 6 — Elbow Method (WCSS, k=1 to k=19)
#
# **What it shows:** Within-Cluster Sum of Squares (WCSS) for k=1 to 19. The elbow — where the rate of decrease sharply slows — indicates the optimal number of clusters.
#
# **Conclusion:** The elbow occurs at **k=2**. The WCSS curve flattens significantly after k=2, meaning additional clusters reduce inertia only marginally while increasing model complexity and reducing interpretability.
#
# ---
#
# ### Plot 7 — Feature Correlation Graph
#
# **What it shows:** A network graph where nodes are the 5 numerical features and edges represent pairwise correlations. Edge thickness scales with absolute correlation value. Green = positive correlation; red dashed = negative correlation.
#
# **Conclusion:** `campaign vs day` (0.17) and `balance vs duration` (implicitly 0.17) are the strongest relationships — the thickest green edges. Most other correlations are weak (< 0.10), confirming the 5 features are largely independent. This independence benefits KMeans, as correlated features would bias the Euclidean distance metric and distort cluster shapes.
#
# ---
#
# ### Overall Clustering Conclusion
#
# KMeans clustering on the 2,896 confirmed subscribers reveals **two distinct behavioural segments**:
#
# | Segment | Profile | Campaign Strategy |
# |---|---|---|
# | **Cluster 0** — High-effort converters | Younger, lower balance, longer calls, more contacts, contacted later in month | Allow more agent time; prioritise early-month contact attempts |
# | **Cluster 1** — Low-effort converters | Older, higher balance, shorter calls, fewer contacts, contacted earlier in month | Prioritise in targeting; high ROI per call |
#
# These segments only emerge after DuckDB filtering to `y = 'yes'` — they are invisible in the full imbalanced dataset. The bank should develop **differentiated contact strategies** for each segment to maximise conversion efficiency.
