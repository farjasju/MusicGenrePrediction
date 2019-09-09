import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import seaborn as sns

le = LabelEncoder()
data = pd.read_csv("./data/data.csv")
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]

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
    projected = pca.fit_transform(dataset.to_numpy())
    print(pca.explained_variance_ratio_)
    return projected

def correlation_matrix(data):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.shape[1]), data.columns, fontsize=10, rotation=90)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.show()

def distribution_plot(data, filename='output'):
    #sns.set(color_codes=True)
    plot = sns.distplot(data)
    fig = plot.get_figure()
    fig.savefig(os.path.join('results',filename + '_distribution.png'))
    plt.clf()

def main():
    # Plotting the distributions
    for var_name in ['tempo', 'beats', 'chroma_stft', 'rmse',
       'spectral_centroid', 'spectral_bandwidth', 'rolloff',
       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
       'mfcc20']:
        distribution_plot(data[var_name], var_name)

    data['label'] = le.fit_transform(data['label'].astype('str'))
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print('\nLabel transformations:', le_name_mapping, '\n')

    print(data.columns)
    print(data.cov())
    print(data.describe())
    print(data.corr())

    # Deleting columns with correlation >= 0.8. The data above is before the removal, and the graph below is after
    correlation_threshold_removal(data, 0.8)

    # Getting the correlation matrix
    correlation_matrix(data)

    # PCA
    projected = pca(x, 2)
    plt.scatter(projected[:, 0], projected[:, 1],
            c=y, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()