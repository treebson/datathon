# +
# Imports

import pandas as pd
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# +
# Read dataframes

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender = pd.read_csv("gender_submission.csv")
# -

pkey = "PassengerId"
response = "Survived"
categoricals = [
    "Pclass", # Ticket class
    "Name", # TODO: feature engineering
    "Sex",
    "Cabin", # TODO: feature engineering
    "Ticket",
    "Embarked"
]
numerics = [
    "Age",
    "SibSp", # of siblings / spouses aboard the Titanic
    "Parch", # of parents / children aboard the Titanic
    "Fare",
]


# +
# Feature engineering

def encode_categoricals(df, categoricals):
    for c in categoricals:
        df[c] = df[c].astype("category").cat.codes
    return df

def inpute_median(df, numerics):
    df[numerics] = df[numerics].fillna(df.median())
    return df

df = encode_categoricals(df, categoricals)
df[numerics] = df[numerics].fillna(df.median())

df


# +
def split(df, seed, split_ratio=0.2):
    features = categoricals + numerics
    df_train, df_test = train_test_split(df, test_size=split_ratio, random_state=seed)
    X_train = df_train[features].values
    y_train = df_train[response].values
    X_test = df_test[features].values
    y_test = df_test[response].values
    return X_train, y_train, X_test, y_test

def train_model(X, y, seed):
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=seed)
    model.fit(X, y)
    return model

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    y_score = y_pred_proba[:,1]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    return(acc, auc)
    
def feature_importance(model, columns):
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": columns,
        "importance": importances
    })
    df_importance = df_importance.sort_values(by="importance", ascending=False)
    return df_importance

def calculate_average_feature_importance(importances, n_bootstraps):
    importances_sorted = [imp.sort_values("feature") for imp in importances]
    values = [0 for _ in range(len(importances_sorted[0]))]
    for imp in importances_sorted:
        imp_vals = imp["importance"].values.tolist()
        for j, val in enumerate(imp_vals):
            values[j] = values[j] + val
    values = [v/n_bootstraps for v in values]
    ave_importance = importances_sorted[0]
    ave_importance.importance = values
    ave_importance = ave_importance.sort_values("importance", ascending=False).reset_index()
    ave_importance = ave_importance[["feature", "importance"]] 
    return ave_importance

def print_progress_bar(iteration, total, prefix="", suffix="", length=30, fill="=", head=">", track="."):
    filled_length = int(length * iteration // total)
    if filled_length == 0:
        bar = track * length
    elif filled_length == 1:
        bar = head + track * (length - 1)
    elif filled_length == length:
        bar = fill * filled_length
    else:
        bar = fill * (filled_length-1) + ">" + "." * (length-filled_length)
    print("\r" + prefix + "[" + bar + "] " + str(iteration) + "/" + str(total), suffix, end = "\r")
    if iteration == total: 
        print()

def bootstrap(df, n_bootstraps):
    print("Training with", n_bootstraps, "bootstraps...")
    accs, aucs, importances = [], [], []
    for i, seed in enumerate(range(n_bootstraps)):
        X_train, y_train, X_test, y_test = split(df, seed)
        model = train_model(X_train, y_train, seed)
        acc, auc = evaluate_model(model, X_test, y_test)
        importance = feature_importance(model, categoricals + numerics)
        accs.append(acc)
        aucs.append(auc)
        importances.append(importance)
        print_progress_bar(i+1, n_bootstraps)
    acc_mean = statistics.mean(accs)
    acc_stdev = statistics.stdev(accs)
    print("\nacc: mean=" + str(acc_mean), "stdev=" + str(acc_stdev))
    auc_mean = statistics.mean(aucs)
    auc_stdev = statistics.stdev(aucs)
    print("auc: mean=" + str(auc_mean), "stdev=" + str(auc_stdev))
    ave_importance = calculate_average_feature_importance(importances, n_bootstraps)
    print(" ")
    print(ave_importance)
        
bootstrap(df, 100)
# -


