import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import seaborn as sns

def correlation_threshold_removal(dataset, threshold):
    '''Removes the columns above a certain level of correlation'''
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

def pca(dataset, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(dataset.to_numpy())
    print(pca.explained_variance_ratio_) 

def correlation_matrix(data):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.shape[1]), data.columns, fontsize=10, rotation=90)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.show()

def distribution_plot(data, filename='output'):
    sns.set(color_codes=True)
    plot = sns.distplot(data)
    fig = plot.get_figure()
    fig.savefig(os.path.join('results',filename + '_distribution.png'))
    plt.clf()
    
def main():

    le = LabelEncoder()
    data = pd.read_csv("./data/data.csv")
    x = data.iloc[:,1:-1]
    y = data.iloc[:,-1]
    
    # Plotting the distributions
    for var_name in ['tempo', 'beats', 'chroma_stft', 'rmse',
       'spectral_centroid', 'spectral_bandwidth', 'rolloff',
       'zero_crossing_rate']:
        distribution_plot(data[var_name], var_name)

    # data['label'] = le.fit_transform(data['label'].astype('str'))
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print('\nLabel transformations:', le_name_mapping, '\n')

    # use pd.concat to join the new columns with your original dataframe
    data = pd.concat([data,pd.get_dummies(data['label'], prefix='label')],axis=1)

    # now drop the original 'country' column (you don't need it anymore)
    data.drop(['label'],axis=1, inplace=True)

    print(data.columns)
    print(data.cov())
    print(data.describe())
    print(data.corr())

    # Deleting columns with correlation >= 0.8. The data above is before the removal, and the graph below is after
    correlation_threshold_removal(data, 0.8)

    # Getting the correlation matrix
    correlation_matrix(data)

    # PCA
    pca(x, 4)

if __name__ == "__main__":
    main()