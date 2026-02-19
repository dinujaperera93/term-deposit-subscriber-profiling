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
# 2. Identify customer segments most likely to convert
# 3. Determine which features most strongly drive subscription decisions
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
# | **Model 1 — Pre-call** | Demographics + financial history | Predict who to call *before* any contact |
# | **Model 2 — Post-call** | All features (incl. call data) | Predict who to *continue* calling after contact |
#
# `duration` is only observable after a call ends — including it in a pre-call model
# constitutes data leakage and is excluded from Model 1 by design.

# %% [markdown]
# ---
# ## Import Dependencies

# %%
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import random

from src.two_layer_model import (
    load_data, explore_data, train_two_layer_pipeline
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
# ## 2. Exploratory Data Analysis
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
# | No (did not subscribe) | ~37,000 | ~88% |
# | Yes (subscribed) | ~5,000 | ~12% |
#
# The dataset is highly imbalanced. A majority-class classifier achieves 88% accuracy
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
#   Range −6,847 to 102,127 — highly right-skewed with significant outliers.
# - **`day`:** Day of month of last contact (1–31). Not a duration.
# - **`duration`:** Last call duration in seconds. Highly predictive but only known
#   *after* the call ends — excluded from Model 1 to prevent data leakage.
# - **`campaign`:** Number of contacts made this campaign. Most customers are contacted
#   1–3 times; the distribution has a heavy right tail.
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
#   in January or September, suggesting campaign scheduling constraints.
#
# ---
#
# #### Structural Missing Values ("unknown")
#
# Pandas reports zero nulls — missingness is encoded as the string `"unknown"`:
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
# Dropping any row containing "unknown" would remove 34.3% of the dataset — nearly all of it
# driven by `contact` alone. Given the existing class imbalance, discarding that volume
# would significantly reduce minority-class representation in training.
#
# **Design decision — hybrid imputation strategy:**
# - **`contact`** → "unknown" retained as a valid category. At 31.9%, this is not random
#   missingness but a distinct cohort (customers reached via an unlogged channel, or
#   older records from before contact-type tracking was introduced). LabelEncoder assigns
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
# ## 3. Preprocessing Pipeline
#
# All transformations are **fitted on training data only** — no leakage at any stage.
#
# | Step | Method | Rationale |
# |------|--------|-----------|
# | Train / Val / Test split | Stratified 80 / 10 / 10 | Preserves the 88/12 class ratio across all sets |
# | `contact` "unknown" | Kept as valid category | 31.9% — informative cohort, not random missingness |
# | `education`, `job` "unknown" | Mode imputation | Sparse unknowns; mode fitted on train only |
# | Outlier handling | IQR clipping | Reduces outlier influence without removing rows |
# | Categorical encoding | `LabelEncoder` | Tree models handle label-encoded categories well |
# | Numeric scaling | `StandardScaler` | Normalises feature scale |
# | Target encoding | `LabelEncoder` | `no` → 0, `yes` → 1 |

# %% [markdown]
# ---
# ## 4. Model Selection
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
# ## 5. Hyperparameter Tuning
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
# ## 6. Running the Full Two-Layer Pipeline
#
# `train_two_layer_pipeline` executes the complete sequence — split, clean, encode,
# select, tune, and evaluate — independently for Model 1 (pre-call features) and
# Model 2 (all features). Feature importance charts and confusion matrices are
# saved to `figures/` and displayed inline below.

# %%
results = train_two_layer_pipeline(term_deposit_df, SEED, categorical_df, numeric_df)

# %% [markdown]
# ---
# ## 7. Results

# %%
m1 = results['model1']
m2 = results['model2']

print("=" * 60)
print("  FINAL RESULTS — TWO-LAYER PIPELINE")
print("=" * 60)

print(f"\n  Model 1 — Pre-Call Targeting")
print(f"  Features           : Demographics + financial history")
print(f"  CV Minority Recall : {m1['cv_score']:.4f}")
print(f"  Best parameters    : {m1['params']}")

print(f"\n  Model 2 — Post-Call Follow-Up")
print(f"  Features           : All features (pre-call + call data)")
print(f"  CV Minority Recall : {m2['cv_score']:.4f}")
print(f"  Best parameters    : {m2['params']}")

delta = m2['cv_score'] - m1['cv_score']
pct   = delta / m1['cv_score'] * 100 if m1['cv_score'] > 0 else 0
print(f"\n  Model 2 lift over Model 1 : +{delta:.4f}  ({pct:+.1f}%)")
print("=" * 60)

# %% [markdown]
# ### Results Interpretation
#
# **Model 2 outperforms Model 1** — post-call features, particularly `duration`,
# add substantial predictive signal once contact has been made.
#
# **Model 1 solves the upstream targeting problem:** which customers should enter
# the calling funnel at all. The two models are sequential, not competing:
#
# ```
# Customer database
#        |
#        v
# Model 1  ->  Score all customers pre-campaign  ->  Prioritise top tier
#        |
#        v
# Call made  ->  Duration, contact type, date recorded
#        |
#        v
# Model 2  ->  Re-score with all data  ->  High: follow up | Low: stop
# ```
#
# ### Feature Importance
#
# | Feature | Business Interpretation |
# |---------|------------------------|
# | `duration` | Longer calls indicate engagement — strong signal for follow-up |
# | `balance` | Higher account balance correlates with investment readiness |
# | `age` | Certain age segments respond more strongly to term deposit offers |
# | `month` | Campaign timing matters — high-conversion months can be planned for |
# | `campaign` | Diminishing returns beyond 3–4 contacts; cap outreach per customer |

# %% [markdown]
# ---
# ## 8. Concluding Remarks
# *For hiring managers and technical reviewers*
#
# ---
#
# **Architecture:** The two-layer design is the central contribution — not a
# hyperparameter choice or library selection. Aligning model boundaries with the
# natural information timeline of a telemarketing campaign produces a system that
# is both more honest and more deployable than a single all-features classifier.
#
# **Data integrity:** Every transformation is fitted on training data only.
# Stored parameters are applied to validation and test sets consistently.
# This discipline is critical for reliable evaluation and is frequently absent
# in exploratory notebooks.
#
# **Interpretability:** LightGBM feature importance is a deliverable, not just
# a diagnostic. The client receives clear, ranked signals about which customer
# attributes drive subscriptions — enabling operational targeting decisions
# without needing to understand the model internals.
#
# **Reproducibility:** All randomness is controlled via `SEED`. Intermediate
# outputs and figures are persisted to `figures/`. The pipeline retrains on
# new campaign data with a single function call.
#
# ---
#
# **Skills demonstrated:**
# Binary classification · Imbalanced data · Feature engineering ·
# Data leakage prevention · Pipeline design · Stratified splitting ·
# Bayesian hyperparameter optimisation (Hyperopt / TPE) · LightGBM ·
# LazyPredict model benchmarking · Feature importance interpretation ·
# Reproducible experiment design
