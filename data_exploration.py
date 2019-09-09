import os

import matplotlib.pyplot as plt
from matplotlib import colors as pltclrs
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import seaborn as sns

def optimal_components(x, y, goal_var: float) -> int:
    lda = LDA(n_components=None)
    X_lda = lda.fit(x, y)

    # Create array of explained variance ratios
    lda_var_ratios = lda.explained_variance_ratio_
    print(lda_var_ratios)

    # Set initial variance explained so far
    total_variance = 0.0
    
    # Set initial number of features
    n_components = 0
    
    # For the explained variance of each feature:
    for explained_variance in lda_var_ratios:
        
        # Add the explained variance to the total
        total_variance += explained_variance
        
        # Add one to the number of components
        n_components += 1
        
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    
    print('The optimal number of components for this dataset is ', n_components, ' components.')
    # Return the number of components
    return n_components

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

def correlation_matrix(data, filename='output'):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.shape[1]), data.columns, fontsize=10, rotation=90)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=10)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.savefig(os.path.join('results', 'correlation matrix ' + filename + '_distribution.png'))

def distribution_plot(data, filename='output'):
    #sns.set(color_codes=True)
    plot = sns.distplot(data)
    fig = plot.get_figure()
    fig.savefig(os.path.join('results',filename + '_distribution.png'))
    plt.clf()

def LDAwithGraphics(x,y,data):
    le = LabelEncoder()
    num_comp = optimal_components(x,y,0.95)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    lda = LDA(n_components=num_comp)
    X_lda = lda.fit_transform(x, y)
    y_label = le.fit_transform(data['label'])
    for i in range(0, num_comp):
        for j in range(0, num_comp):
            if (i != j):
                plt.figure(figsize=(10,8))
                for lab, col in zip(('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'), ['red','green','blue','purple','gray','brown','orange','cyan','pink','olive']):
                    plt.scatter(
                        X_lda[y == lab,i],
                        X_lda[y == lab,j],
                        c=col,
                        label=lab,
                        alpha=0.7
                        )
                plt.xlabel('component'+str(i))
                plt.ylabel('component'+str(j))
                plt.title('LDA')
                plt.legend()
                plt.savefig(os.path.join('results', 'projection_matrix_'+str(i)+'_'+str(j)+'.png'))
                plt.clf()


    # plt.figure(figsize=(10,8))
    # for lab, col in zip(('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'), ['red','green','blue','purple','gray','brown','orange','cyan','pink','olive']):
    #     plt.scatter(
    #         X_lda[y == lab,0],
    #         X_lda[y == lab,1],
    #         c=col,
    #         label=lab,
    #         alpha=0.7
    #         )
    # plt.xlabel('component 1')
    # plt.ylabel('component 2')
    # plt.title('LDA - First axis')
    # plt.legend()
    # plt.savefig(os.path.join('results', 'projection_matrix_first_axis.png'))
    # plt.clf()

    # plt.figure(figsize=(10,8))
    # for lab, col in zip(('blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'), ['red','green','blue','purple','gray','brown','orange','cyan','pink','olive']):
    #     plt.scatter(
    #         X_lda[y == lab,0],
    #         X_lda[y == lab,2],
    #         c=col,
    #         label=lab,
    #         alpha=0.7
    #         )
    # plt.xlabel('component 1')
    # plt.ylabel('component 3')
    # plt.title('LDA - Second axis')
    # plt.legend()
    # plt.savefig(os.path.join('results', 'projection_matrix_second_axis.png'))
    # plt.clf()

    # for i in range(0, num_comp-1):
    #     plt.scatter(
    #     X_lda[:,i],
    #     X_lda[:,i+1],
    #     c=y_label,
    #     cmap='rainbow',
    #     alpha=0.7,
    #     edgecolors='b'
    #     )
    #     plt.savefig(os.path.join('results', 'projection matrix ' + str(i) + '.png'))

    return lda

def main():
    le = LabelEncoder()
    data = pd.read_csv("./data/data.csv")
    data.drop(['filename'],axis=1, inplace=True)
    
    x = data.iloc[:,1:-1].values
    y = data.iloc[:,-1].values

    ##Outlier removal - using interquantil distance, because our variables follow a mostly normal distribution
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # print("IQR:", IQR)

    data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
    # print('Shape of dataset before outlier removal: ', data.shape)
    # print('Shape of dataset before outlier removal: ', data_out.shape)

    x_out = data_out.iloc[:,1:-1].values
    y_out = data_out.iloc[:,-1].values

    ##########################

    # X_train = lda.fit_transform(X_train, y_train)
    # X_test = lda.transform(X_test)

    # classifier = RandomForestClassifier(max_depth=2, random_state=0)

    # classifier.fit(X_train, y_train)
    # y_pred = classifier.predict(X_test)

    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # print('Accuracy' + str(accuracy_score(y_test, y_pred)))

    # Plotting the distributions
    for var_name in ['tempo', 'beats', 'chroma_stft', 'rmse',
       'spectral_centroid', 'spectral_bandwidth', 'rolloff',
       'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
       'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
       'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
       'mfcc20']:
        distribution_plot(data[var_name], var_name)

    comp_x = [1,2,3,4,5,6,7]
    comp_h = [0.39899075,0.2409215,0.11481741,0.10215124,0.04666102,0.03713209, 0.03028714]
    plt.bar(comp_x, height=comp_h)
    plt.show()

    lda = LDAwithGraphics(x_out,y_out, data_out)
    # LDAwithGraphics(x, y, data)
    print(lda.explained_variance_ratio_)

    ##Label encoding
    # data['label'] = le.fit_transform(data['label'].astype('str'))
    # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # print('\nLabel transformations:', le_name_mapping, '\n')

    ##One hot encoding
    # use pd.concat to join the new columns with your original dataframe
    data = pd.concat([data,pd.get_dummies(data['label'], prefix='label')],axis=1)

    # now drop the original 'country' column (you don't need it anymore)
    data.drop(['label'],axis=1, inplace=True)

    print(data.columns)
    print(data.cov())
    print(data.describe())
    print(data.corr())

    ####################

    correlation_matrix(data, 'before threshold removal')

    # Deleting columns with correlation >= 0.8. The data above is before the removal, and the graph below is after
    correlation_threshold_removal(data, 0.8)

    # Getting the correlation matrix
    correlation_matrix(data, 'after threshold removal')

    # PCA
    # projected = pca(x, 2)
    # label = [x//100 for x in range(1000)]
    # colors = ['red','green','blue','purple','gray','brown','orange','cyan','pink','olive']
    # plt.scatter(projected[:, 0], projected[:, 1],
    #         c=label, edgecolor='none', alpha=0.5,
    #         cmap=pltclrs.ListedColormap(colors))
    # cb = plt.colorbar()
    # loc = np.arange(0,max(label),max(label)/float(len(colors)))
    # cb.set_ticks(loc)
    # cb.set_ticklabels(colors)
    # plt.xlabel('component 1')
    # plt.ylabel('component 2')
    # plt.show()

if __name__ == "__main__":
    main()