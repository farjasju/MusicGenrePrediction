import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sn

def load_data():
    data = pd.read_csv("./data/data.csv")
    print(data.head())

    # Data matrix
    X = data.to_numpy()[:, 1:-1]

    # Label matrix
    y = data.to_numpy()[:, -1]
    y = y.flatten()

    # Transform the problem in a binary classification problem
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
    return X_train, X_test, y_train, y_test, lb

def evaluate(y_pred, y_test, lb):
    print("Average precision score:", str(metrics.average_precision_score(y_pred, y_test)))
    print("Accuracy score:", str(metrics.accuracy_score(y_pred, y_test)))
    print("F1 score:", str(metrics.f1_score(y_pred, y_test, average='micro')))
    print("Precision score:", str(metrics.precision_score(y_pred, y_test, average='micro')))
    print("Recall score:", str(metrics.recall_score(y_pred, y_test, average='micro')))
    print("ROC AUC score:", str(metrics.roc_auc_score(y_pred, y_test, average='micro')))
    
    # ROC Curve ?

    cm = metrics.confusion_matrix(y_pred.argmax(1), y_test.argmax(1))
    df_cm = pd.DataFrame(cm, index = [i for i in lb.classes_],
                  columns = [i for i in lb.classes_])
    plt.figure(figsize = (10,7))
    ax = sn.heatmap(df_cm, annot=True)
    ax.set_ylim(10.0, 0)
    plt.show()

def main():
    # Loading the dataset
    X_train, X_test, y_train, y_test, lb = load_data()

    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_test_labels = lb.inverse_transform(y_test)
    y_pred_labels = lb.inverse_transform(y_pred)

    evaluate(y_pred, y_test, lb)

if __name__ == '__main__':
    main()