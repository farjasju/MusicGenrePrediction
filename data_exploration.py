import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data = pd.read_csv("./data/data.csv")

#Correlation threshold removal function
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
    

    print(dataset, col_corr)

data['label'] = le.fit_transform(data['label'].astype('str'))
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print('\nLabel transformations:', le_name_mapping, '\n')

print(data.columns)
print(data.cov())
print(data.describe())
print(data.corr())

#Deleting columns with correlation >= 0.8. The data above is before the removal, and the graph below is after
correlation(data, 0.8)

#Geting the correlation matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(data.corr(), fignum=f.number)
plt.xticks(range(data.shape[1]), data.columns, fontsize=10, rotation=90)
plt.yticks(range(data.shape[1]), data.columns, fontsize=10)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=12)

plt.show()