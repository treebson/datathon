# +
# Imports

import pandas as pd
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
# Train/test split

seed = 42
split_ratio = 0.2

df_train, df_test = train_test_split(
    df, 
    test_size=split_ratio, 
    random_state=seed
)


# +
def train_model(X, y, seed):
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=seed)
    model.fit(X, y)
    return model

X_train = df_train[numerics + categoricals].values
y_train = df_train[response]
model = train_model(X_train, y_train, seed)


# +
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    y_score = y_pred_proba[:,1]
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    print("acc:", accuracy)
    print("auc:", auc)
    
X_test = df_test[numerics + categoricals].values
y_test = df_test[response].values
evaluate_model(model, X_test, y_test)


# +
def feature_importance(model, columns):
    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        "feature": columns,
        "importance": importances
    })
    df_importance = df_importance.sort_values(by="importance", ascending=False)
    return df_importance
    
df_importance = feature_importance(model, numerics + categoricals)
print(df_importance)
# -


