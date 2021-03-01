import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from genetic_selection import GeneticSelectionCV
from featureFinder import find
from sklearn.model_selection import cross_val_score


df = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")
total_feats = list(df.columns)
print(total_feats)


total_feats = total_feats[:-1]
target = df.columns[-1]

catcols = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
# for c in catcols:
#   D[c] = D[c].astype("category")
numcols = set(total_feats) - set(catcols)
numcols = list(numcols)



for col in numcols: 
  scaler = MinMaxScaler()
  df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))

print(df.head())

#for feat in total_feats:
 #   scaler = MinMaxScaler()
  #  df[feat] = scaler.fit_transform(df[feat].values.reshape(-1,1))

#df.iloc[:,-1] = df.iloc[:,-1].astype(np.int64)
print(df.head)


prefs = find(df)

new = []

for i in range(prefs.shape[0]):
    if prefs[i] == True:
        new.append(total_feats[i])

print(new)
X = df[new]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state= 46)

clf = RandomForestClassifier(max_depth = 3)
rskf = RepeatedStratifiedKFold(n_repeats = 10, n_splits = 10)
X_prime = df[new]; y_prime = df.iloc[:,-1]
print(np.mean(cross_val_score(clf, X_prime, y_prime, cv = rskf, scoring = "roc_auc")))


