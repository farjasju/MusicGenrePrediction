import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("./data/data.csv")
print(data.head())

# Data matrix
X = data.to_numpy()[:, 1:-1]

# Label matrix
y = data.to_numpy()[:, -1]
y = y.flatten()

# transformer en un problème de classification binaire
lb = preprocessing.LabelBinarizer()
lb.fit(y)
print(lb.classes_)
y_binarized = lb.transform(y)

# Creating the Training and Test set from data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_binarized, test_size = 0.25, random_state = 21)

# Feature Scaling
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_test_labels = lb.inverse_transform(y_test)
y_pred_labels = lb.inverse_transform(y_pred)

for x in range(100):
    print("ORIGINAL:", y_test_labels[x], "PREDICTED:", y_pred_labels[x])

print(metrics.average_precision_score(y_pred, y_test))
print(metrics.accuracy_score(y_pred, y_test))
print(metrics.f1_score(y_pred, y_test, average='micro'))
print(metrics.precision_score(y_pred, y_test, average='micro'))
print(metrics.recall_score(y_pred, y_test, average='micro'))