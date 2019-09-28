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
df

# +
# Feature engineering

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

def encode_categoricals(df, categoricals):
    for c in categoricals:
        df[c] = df[c].astype("category").cat.codes
    return df

def inpute_median(df, numerics):
    df[numerics] = df[numerics].fillna(df.median())
    return df

def engineer_features(df):
    df = encode_categoricals(df, categoricals)
    df[numerics] = df[numerics].fillna(df.median())
    return df

df = engineer_features(df)
df


# +
def split(df, seed, split_ratio=0.2):
    df_train, df_test = train_test_split(df, test_size=split_ratio, random_state=seed)
    return df_train, df_test

def extract_features(df):
    features = categoricals + numerics
    X = df[features].values
    return X

def extract_response(df):
    y = df[response].values
    return y

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
    accs, aucs, importances, models = [], [], [], []
    for i, seed in enumerate(range(n_bootstraps)):
        df_train, df_test = split(df, seed)
        X_train = extract_features(df_train)
        y_train = extract_response(df_train)
        X_test = extract_features(df_test)
        y_test = extract_response(df_test)
        model = train_model(X_train, y_train, seed)
        acc, auc = evaluate_model(model, X_test, y_test)
        importance = feature_importance(model, categoricals + numerics)
        accs.append(acc)
        aucs.append(auc)
        importances.append(importance)
        models.append(model)
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
    best_index = [i for i, auc in enumerate(aucs) if auc == max(aucs)][0]
    best_model = models[best_index]
    return best_model
        
best_model = bootstrap(df, 100)
# + {}
def score_test(best_model):
    df_test = pd.read_csv("test.csv")
    df_test = engineer_features(df_test)
    X_test = extract_features(df_test)
    survived = best_model.predict(X_test)
    passengers = df_test["PassengerId"].values
    submission = pd.DataFrame({
        "PassengerId": passengers,
        "Survived": survived
    })
    return submission
    
submission = score_test(best_model)
submission.to_csv("submission.csv", index=False)
submission
# -



