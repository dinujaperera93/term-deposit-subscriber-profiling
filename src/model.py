import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, make_scorer, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def load_data(filepath):
    return pd.read_csv(filepath)

def categorical_counts(value_counts_):
    # display(HTML(pd.DataFrame(value_counts_).to_html()))
    print(value_counts_.to_string())

def explore_data(df): 
    print(pd.concat([df.head(5), df.tail(5)]))
    print(f"Shape of the dataset : {df.shape}")
    print(f"Columns of the dataset : {df.columns}")
    print("Number of records in the dataframe; {}\nNumber of duplicate records: {}".format(df.shape[0],df.duplicated().sum()))
    df.info()
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    categorical_df = df.select_dtypes(include=["object"])
    
    print("\nNumerical Features Summary:\n")
    print(numeric_df.describe())
    print("\nCategorical Features Summary:\n")
    print(categorical_df.describe())
    
    plt.figure()
    ax = sns.countplot(x=df["y"])
    total = len(df["y"])
    for p in ax.patches:
        count = p.get_height()
        percentage = 100 * count / total
        ax.annotate(f'{count}\n({percentage:.0f}%)', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center')
    plt.title("Target Distribution")
    plt.xlabel("Term deposit (no / yes)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(FIGURES_DIR / "target_distribution.png", bbox_inches="tight")
    plt.show()
    plt.close()
    
    return numeric_df, categorical_df
    
def split_data(df, target, seed, test_size=0.2, val_size = 0.2):
    X = df.loc[:, df.columns != target]
    y = df[target].values.ravel()

    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=(test_size + val_size), random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp, test_size= test_size, random_state=seed, stratify=y_temp)
    print(f"Size of the training set: {X_train.shape[0]}\nSize of the validation set: {X_val.shape[0]}\nSize of the testing set: {X_test.shape[0]}")

    return X_train, X_val, X_test, y_train, y_val, y_test
    
def EDA(X_train, y_train, X_val, y_val, categorical_df, numeric_df):
    
    cat_cols = categorical_df.columns.tolist()
    num_cols = numeric_df.columns.tolist()
    # Combine train + val
    X_tv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_tv = pd.Series(list(y_train) + list(y_val), name="y")
    df_tv = pd.concat([X_tv, y_tv], axis=1)
    
    # Value counts for categorical columns
    print("\nCategorical Value Counts (Train + Val)\n")
    for col in cat_cols:
        categorical_counts(df_tv[col].value_counts())
        
    for col in cat_cols:
        plt.figure()
        sns.countplot(x=df_tv[col], hue=y_tv)
        plt.title(f"{col}")
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.savefig(FIGURES_DIR / f"feature_{col}.png", bbox_inches='tight') 
        plt.show()
        plt.close()

    corr = df_tv[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Correlation Heatmap (Train + Val)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "corr_heatmap_train_val.png", bbox_inches="tight")
    plt.show()
    plt.close()

    for col in num_cols:
        s = df_tv[col]
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
        plt.savefig(FIGURES_DIR / f"hist_{col}.png", bbox_inches="tight")
        plt.show()
        plt.close()
    
    return df_tv
    
def data_cleaning(df, categorical_df, numeric_df):
    df = df.copy()

    # Store train-fitted parameters for test set
    cat_mode = {}
    num_bounds = {}  # {col: (lower, upper)}

    for col in categorical_df.columns:
        df[col] = df[col].replace("unknown", pd.NA)
        mode_value = df[col].mode(dropna=True)[0]
        cat_mode[col] = mode_value
        df[col] = df[col].fillna(mode_value)

    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        num_bounds[col] = (lower, upper)
        df[col] = df[col].clip(lower, upper)

    return df, cat_mode, num_bounds


def encode_data(X_train, X_val, y_train, y_val, categorical_df, numeric_df):
    X_train = X_train.copy()
    X_val = X_val.copy()
    
    cat_cols = [c for c in categorical_df.columns.tolist() if c != 'y']
    num_cols = numeric_df.columns.tolist()
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = le.transform(X_val[col])
        le_dict[col] = le
    
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    
    le_y = LabelEncoder()
    y_train = le_y.fit_transform(y_train)
    y_val = le_y.transform(y_val)
    
    return X_train, X_val, y_train, y_val, le_dict, scaler, le_y
    
        
def select_model(X_train, X_val, y_train, y_val):
    # Minority class is 1
    def minority_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, pos_label=1)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=minority_recall)
    models, predictions = clf.fit(X_train, X_val, y_train, y_val)

    print(f"\nBest model for minority class: {models['minority_recall'].idxmax()}")
    return models, predictions

def compare_ensembles(X_train, y_train, X_val, y_val, seed, cv=5):
    minority_recall = make_scorer(recall_score, pos_label=1)
    
    # Combine train and val for cross-validation
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = np.concatenate([y_train, y_val])
    
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(random_state=seed, class_weight="balanced"),
        "LGBM": LGBMClassifier(random_state=seed, verbose=-1, class_weight="balanced"),
        "SVM": SVC(random_state=seed, class_weight="balanced", probability=True),
    }

    base = [(k.lower().replace(" ", "_"), v) for k, v in models.items()]
    models["Voting"] = VotingClassifier(estimators=base, voting="soft")
    models["Stacking"] = StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=2000), cv=5)
    
    results = []
    fitted_models = {}
    
    for name, model in models.items():
        mrec = cross_val_score(model, X_combined, y_combined, cv=cv, scoring=minority_recall).mean()
        model.fit(X_combined, y_combined)
        fitted_models[name] = model
        results.append({"Model": name, "Minority_Recall": round(mrec, 4)})
    
    results_df = pd.DataFrame(results).sort_values("Minority_Recall", ascending=False)
   
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=results_df, x="Model", y="Minority_Recall")
    ax.bar_label(ax.containers[0], fmt='%.2f')
    plt.title("Model Comparison - Minority Recall")
    plt.ylabel("Minority Recall")
    plt.grid(True, axis='y')
    plt.savefig(FIGURES_DIR / "model_comparison.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    return fitted_models, results_df

def tune_hyperparameters(X_train, y_train, X_val, y_val, seed):
    np.random.seed(seed)
    minority_recall = make_scorer(recall_score, pos_label=1)
    
    # Combine train and val
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = np.concatenate([y_train, y_val])

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
        params = {
            'n_estimators': int(params['n_estimators']),
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'num_leaves': int(params['num_leaves']),
            'min_child_samples': int(params['min_child_samples']),
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
        }
        model = LGBMClassifier(**params, random_state=seed, verbose=-1, class_weight='balanced')
        score = cross_val_score(model, X_combined, y_combined, cv=5, scoring=minority_recall).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    rstate = np.random.default_rng(seed)
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=rstate)
    
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'num_leaves': int(best['num_leaves']),
        'min_child_samples': int(best['min_child_samples']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
    }
    
    best_model = LGBMClassifier(**best_params, random_state=seed, verbose=-1, class_weight='balanced')
    best_model.fit(X_combined, y_combined)
    best_score = -min(t['result']['loss'] for t in trials.trials)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {round(best_score, 2)}")
    
    return best_model, best_params, best_score

def important_features(X_train, model):
    importances = model.feature_importances_
    
    feature_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': np.round(importances, 6)
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=feature_df, x='Importance', y='Feature')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", bbox_inches='tight')
    plt.show()
    plt.close()
    
    return feature_df


def evaluate_model(model,X_test,y_test,categorical_df,numeric_df,cat_mode,num_bounds,le_dict,scaler,le_y):
    X_test = X_test.copy()
    cat_cols = [c for c in categorical_df.columns.tolist() if c != "y"]
    num_cols = numeric_df.columns.tolist()

    # Cleaning on X_test using train-fitted params
    for col in cat_cols:
        X_test[col] = X_test[col].replace("unknown", pd.NA)
        X_test[col] = X_test[col].fillna(cat_mode[col])

    for col in num_cols:
        lower, upper = num_bounds[col]
        X_test[col] = X_test[col].clip(lower, upper)

    # Encoding categorical using train-fitted label encoders
    for col in cat_cols:
        X_test[col] = le_dict[col].transform(X_test[col])

    # Scaling numeric using train-fitted scaler
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Encode y_test
    y_test_enc = le_y.transform(y_test)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test_enc, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.show()
    plt.close()
    clf_report = classification_report(y_test_enc, y_pred, target_names=le_y.classes_)
    return clf_report
