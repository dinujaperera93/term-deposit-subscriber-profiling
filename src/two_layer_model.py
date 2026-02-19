import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (recall_score, make_scorer, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from lazypredict.Supervised import LazyClassifier
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PRE_CALL_FEATURES = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan']
POST_CALL_FEATURES = ['contact', 'day', 'month', 'duration', 'campaign']

def load_data(filepath):
    return pd.read_csv(filepath)

def get_feature_sets(df):
    pre_call_cols = [c for c in PRE_CALL_FEATURES if c in df.columns]
    post_call_cols = [c for c in POST_CALL_FEATURES if c in df.columns]
    return pre_call_cols, post_call_cols

def explore_data(df):
    print(pd.concat([df.head(5), df.tail(5)]))
    print(f"Number of records: {df.shape[0]}\n Number of Columns : {df.shape[1]}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    df.info()

    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    categorical_df = df.select_dtypes(include=['object'])
    
    numeric_cols = numeric_df.columns.tolist()
    categorical_cols = categorical_df.columns.tolist()
    
    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    print("\nNumerical Summary:\n", numeric_df.describe().round(2).T)
    print("\nCategorical Summary:\n", categorical_df.describe())

    # Target distribution
    target_counts = df['y'].value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c','#2ecc71']
    bars = ax.bar(target_counts.index, target_counts.values, color=colors)
    for bar, count in zip(bars, target_counts.values):
        pct = 100 * count / len(df)
        ax.annotate(f'{count:,}\n({pct:.1f}%)', 
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Term Deposit Subscription', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Target Variable Distribution (y)', fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "target_distribution.png", bbox_inches="tight")
    plt.show()
    plt.close()

    # Categorical feature distributions
    categorical_cols = [c for c in categorical_df.columns if c != 'y']
    for col in categorical_cols:
        print(f"\n--- {col} ---")
        value_counts = df[col].value_counts()
        print(value_counts)
        plt.figure()
        sns.countplot(x=df[col], hue=df['y'])
        plt.title(col)
        plt.ylabel('Count')
        plt.xticks(rotation=75)
        plt.grid(True)
        plt.savefig(FIGURES_DIR / f"feature_{col}.png", bbox_inches='tight')
        plt.show()
        plt.close()
        
    print("\nUnknown value counts:")
    unk = {col: (df[col] == 'unknown').sum() for col in categorical_cols if (df[col] == 'unknown').any()}
    for col, n in unk.items():
        print(f"  {col}: {n:,} ({100 * n / len(df):.1f}%)")

    cols_with_unk = list(unk.keys())
    if len(cols_with_unk) >= 2:
        print("\nPairwise overlap:")
        for i in range(len(cols_with_unk)):
            for j in range(i + 1, len(cols_with_unk)):
                a, b = cols_with_unk[i], cols_with_unk[j]
                n = ((df[a] == 'unknown') & (df[b] == 'unknown')).sum()
                print(f"  {a} & {b}: {n:,}")

    if len(cols_with_unk) >= 3:
        a, b, c = cols_with_unk[:3]
        n_all = ((df[a] == 'unknown') & (df[b] == 'unknown') & (df[c] == 'unknown')).sum()
        print(f"  all three: {n_all:,}")

    n_union = (df[cols_with_unk] == 'unknown').any(axis=1).sum()
    print(f"\nRows lost if any unknown dropped: {n_union:,} ({100 * n_union / len(df):.1f}%)")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="Blues")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "corr_heatmap.png", bbox_inches="tight")
    plt.show()
    plt.close()

    # Histograms
    for col in numeric_df.columns:
        s = df[col]
        plt.figure(figsize=(8, 4))
        plt.hist(s, bins=30)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        mn, av, mx = s.min(), s.mean(), s.max()
        plt.axvline(mn, linewidth=1)
        plt.axvline(av, linewidth=1)
        plt.axvline(mx, linewidth=1)
        ax = plt.gca()
        ax.text(0.02, 0.98, f"min: {mn:.3f}\nmean: {av:.3f}\nmax: {mx:.3f}",
                transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(FIGURES_DIR / f"hist_{col}.png", bbox_inches="tight")
        plt.show()
        plt.close()
        
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
    if col == 'age':
        print(f"age: min={min_val}, max={max_val}  (reasonable)")
    elif col == 'balance':
        neg_count = (df[col] < 0).sum()
        print(f"balance: min={min_val}, max={max_val}, negative values={neg_count:,} (overdrafts - normal)")
    elif col == 'day':
        print(f"day: min={min_val}, max={max_val} ✓ (days of month)")
    elif col == 'duration':
        print(f"duration: min={min_val}, max={max_val} ✓ (call duration in seconds)")
    elif col == 'campaign':
        print(f"campaign: min={min_val}, max={max_val} (# contacts this campaign)")

    return numeric_df, categorical_df

def split_data(df, target, seed, test_size=0.1, val_size=0.1):
    X = df.loc[:, df.columns != target]
    y = df[target].values.ravel()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=seed, stratify=y_temp)
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def data_cleaning(X_train, X_val, categorical_df, numeric_df):
    cat_cols = [c for c in categorical_df.columns if c != 'y']
    num_cols = numeric_df.columns.tolist()

    # df['day_month'] = df['day'].astype(str) + '_' + df['month']
    # df = df.drop(['day', 'month'], axis=1)

    # Fit on train
    X_train_cleaned = X_train.copy()
    cat_mode = {}
    num_bounds = {}

    # contact "unknown" is kept as a valid category (31.9% — informative, not noise)
    # education and job "unknown" are mode-imputed (sparse, not a meaningful segment)
    impute_cols = [c for c in cat_cols if c != 'contact']
    for col in impute_cols:
        X_train_cleaned[col] = X_train_cleaned[col].replace("unknown", pd.NA)
        mode_value = X_train_cleaned[col].mode(dropna=True)[0]
        cat_mode[col] = mode_value
        X_train_cleaned[col] = X_train_cleaned[col].fillna(mode_value)

    for col in num_cols:
        Q1, Q3 = X_train_cleaned[col].quantile(0.25), X_train_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        num_bounds[col] = (lower, upper)
        X_train_cleaned[col] = X_train_cleaned[col].clip(lower, upper)

    # Apply to val
    X_val_cleaned = X_val.copy()
    for col, mode_val in cat_mode.items():
        X_val_cleaned[col] = X_val_cleaned[col].replace("unknown", pd.NA).fillna(mode_val)
    for col, (lower, upper) in num_bounds.items():
        X_val_cleaned[col] = X_val_cleaned[col].clip(lower, upper)
    
    print(cat_mode, num_bounds)

    return X_train_cleaned, X_val_cleaned, cat_mode, num_bounds, cat_cols, num_cols

def encode_data(X_train_cleaned, X_val_cleaned, y_train, y_val, cat_cols, num_cols):
    X_train_enc, X_val_enc = X_train_cleaned.copy(), X_val_cleaned.copy()
    le_dict = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train_enc[col])
        X_val_enc[col] = le.transform(X_val_enc[col])
        le_dict[col] = le

    scaler = StandardScaler()
    X_train_enc[num_cols] = scaler.fit_transform(X_train_enc[num_cols])
    X_val_enc[num_cols] = scaler.transform(X_val_enc[num_cols])

    le_y = LabelEncoder()
    y_train_enc = le_y.fit_transform(y_train)
    y_val_enc = le_y.transform(y_val)

    return X_train_enc, X_val_enc, y_train_enc, y_val_enc, le_dict, scaler, le_y

def select_model(X_train_enc, X_val_enc, y_train_enc, y_val_enc):
    def minority_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=1)
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=minority_recall)
    models, predictions = clf.fit(X_train_enc, X_val_enc, y_train_enc, y_val_enc)
    print(f"\nBest model for minority recall: {models['minority_recall'].idxmax()}")
    return models, predictions

def tune_hyperparameters(X_train_enc, y_train_enc, X_val_enc, y_val_enc, seed):
    np.random.seed(seed)
    minority_recall = make_scorer(recall_score, pos_label=1)
    X_combined = pd.concat([X_train_enc, X_val_enc], axis=0).reset_index(drop=True)
    y_combined = np.concatenate([y_train_enc, y_val_enc])

    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 500, 25),
        'max_depth': hp.quniform('max_depth', 2, 8, 1),
        'learning_rate': hp.uniform('learning_rate', 0.05, 0.2),
        'num_leaves': hp.quniform('num_leaves', 5, 31, 5),
        'min_child_samples': hp.quniform('min_child_samples', 10, 30, 5),
        'subsample': hp.uniform('subsample', 0.7, 0.9),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.9),
    }

    def objective(params):
        p = {k: int(v) if k in ('n_estimators','max_depth','num_leaves','min_child_samples') else v
             for k, v in params.items()}
        model = LGBMClassifier(**p, random_state=seed, verbose=-1, class_weight='balanced')
        score = cross_val_score(model, X_combined, y_combined, cv=5, scoring=minority_recall).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10,
                trials=trials, rstate=np.random.default_rng(seed))

    best_params = {k: int(v) if k in ('n_estimators','max_depth','num_leaves','min_child_samples') else v
                   for k, v in best.items()}
    best_model = LGBMClassifier(**best_params, random_state=seed, verbose=-1, class_weight='balanced')
    best_model.fit(X_combined, y_combined)
    best_score = -min(t['result']['loss'] for t in trials.trials)
    print(f"Best Params: {best_params}")
    print(f"Best Minority Recall (CV): {best_score:.4f}")
    return best_model, best_params, best_score

def feature_importance(X_train, model):
    feat_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.round(model.feature_importances_, 6)
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_df, x='Importance', y='Feature')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    return feat_df


def evaluate_model(model, X_test, y_test, le_dict, scaler, le_y,
                   cat_cols, num_cols, cat_mode, num_bounds, cols, label=""):
    # Evaluate supervised model on test set
    for col, mode_val in cat_mode.items():  # contact not in cat_mode — its "unknown" is kept
        X_test[col] = X_test[col].replace("unknown", pd.NA).fillna(mode_val)
    for col, (lo, hi) in num_bounds.items():
        X_test[col] = X_test[col].clip(lo, hi)
    for col in cat_cols:
        X_test[col] = le_dict[col].transform(X_test[col])
    
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    y_enc = le_y.transform(y_test)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_enc, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_).plot()
    plt.title(f"Confusion Matrix — {label}")
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{label}.png", bbox_inches="tight")
    plt.show()
    plt.close()
    return classification_report(y_enc, y_pred, target_names=le_y.classes_)

def train_two_layer_pipeline(df, seed, categorical_df, numeric_df):
    pre_call_cols, post_call_cols = get_feature_sets(df)
    all_cols = pre_call_cols + post_call_cols

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target="y", seed=seed)
    X_train_cleaned, X_val_cleaned, cat_mode, num_bounds, cat_cols, num_cols = data_cleaning(X_train, X_val, categorical_df, numeric_df)

    results = {}

    # Model 1: Pre-call features (Who to call)
    pre_cat = categorical_df[[c for c in pre_call_cols if c in categorical_df.columns]]
    pre_num = numeric_df[[c for c in pre_call_cols if c in numeric_df.columns]]

    X_tr_m1, X_va_m1, y_tr_m1, y_va_m1, le1, sc1, le_y1 = encode_data(
        X_train_cleaned[pre_call_cols], X_val_cleaned[pre_call_cols],
        y_train, y_val, pre_cat, pre_num)

    models_df1, _ = select_model(X_tr_m1, X_va_m1, y_tr_m1, y_va_m1)
    print(models_df1)

    model1, params1, score1 = tune_hyperparameters(X_tr_m1, y_tr_m1, X_va_m1, y_va_m1, seed)

    feat1 = feature_importance(X_tr_m1, model1)
    print("\nModel 1 Feature Importance:\n", feat1)

    report1 = evaluate_model(model1, X_test, y_test, le1, sc1, le_y1,
                             pre_cat, pre_num, cat_mode, num_bounds, pre_call_cols, "Model1")
    print(f"\nModel 1 Test Performance:\n{report1}")

    results['model1'] = {'model': model1, 'params': params1, 'cv_score': score1,
                         'features': feat1, 'report': report1}

    # Model 2: All features (Who to continue calling)
    all_cat = categorical_df[[c for c in all_cols if c in categorical_df.columns]]
    all_num = numeric_df[[c for c in all_cols if c in numeric_df.columns]]

    X_tr_m2, X_va_m2, y_tr_m2, y_va_m2, le2, sc2, le_y2 = encode_data(
        X_train_cleaned[all_cols], X_val_cleaned[all_cols],
        y_train, y_val, all_cat, all_num)

    models_df2, _ = select_model(X_tr_m2, X_va_m2, y_tr_m2, y_va_m2)
    print(models_df2)

    model2, params2, score2 = tune_hyperparameters(X_tr_m2, y_tr_m2, X_va_m2, y_va_m2, seed)

    feat2 = feature_importance(X_tr_m2, model2)
    print("\nModel 2 Feature Importance:\n", feat2)

    report2 = evaluate_model(model2, X_test, y_test, le2, sc2, le_y2,
                             all_cat, all_num, cat_mode, num_bounds, all_cols, "Model2")
    print(f"\nModel 2 Test Performance:\n{report2}")

    results['model2'] = {'model': model2, 'params': params2, 'cv_score': score2,
                         'features': feat2, 'report': report2}

    return results
