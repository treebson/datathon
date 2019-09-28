# +
# Imports
import jupytext
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# +
# Read dataframes

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#+
# Name Features

#Function to get title from name

def getTitle(name_col):
    
    s1 = pd.DataFrame(name_col.str.split(",", n=1, expand = True))
    s1.columns = ["Surname", "Other"]

    s2 = pd.DataFrame(s1["Other"].str.split(".", n = 1, expand = True))
    s2.columns = ["Title", "Other"]
    
    return s2["Title"] 
