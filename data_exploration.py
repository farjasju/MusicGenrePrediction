import os

import matplotlib.pyplot as plt
from matplotlib import colors as pltclrs
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from scipy.spatial.distance import pdist, squareform

import seaborn as sns

def pca(x, y, num_comp):
    pca = PCA(n_components=num_comp)
    projected = pca.fit_transform(x, y)
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

def scatter_plot(data):
    pd.plotting.scatter_matrix(data, diagonal="kde")
    plt.tight_layout()
    plt.show()

def main():
    data = pd.read_csv("./data/data.csv")
    data.drop(['filename'],axis=1, inplace=True)
    
    X = data.iloc[:,1:-1].values
    y = data.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    lda = LDA()
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)

    # scatter_plot(pd.DataFrame(X_train)) ##scatterplot to see the LDA

## classifier 1 - random forest
    print('Random forest\n')
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 2 - naive bayes
    print('Naive Bayes:\n')
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 3 - logistic regression
    print('Logistic regression:\n')
    logisticRegr = LogisticRegression(solver='sag', multi_class='auto')
    y_pred = logisticRegr.fit(X_train, y_train).predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 4 - linear SVM
    print('Linear SVM:\n')
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 5 - SVC poly kernel degree 2
    print('SVC poly kernel degree 2:\n')
    svclassifier = SVC(kernel='poly', degree=2, gamma="scale")
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 6 - SVC poly kernel degree 4
    print('SVC poly kernel degree 4:\n')
    svclassifier = SVC(kernel='poly', degree=4, gamma="scale")
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 7 - SVC gaussian kernel
    print('SVC gaussian kernel -- best so far:\n')
    svclassifier = SVC(kernel='rbf', gamma="scale", random_state=0)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 8 - SVC sigmoid kernel
    print('SVC sigmoid kernel:\n')
    svclassifier = SVC(kernel='sigmoid', gamma="scale")
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 9 - Decision tree
    print('Decision tree:\n')
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 10 - KNN 5 neighbors
    print('KNN 5 neighbors:\n')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 11 - KNN 7 neighbors
    print('KNN 7 neighbors --second best:\n')
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 12 - KNN 9 neighbors
    print('KNN 8 neighbors:\n')
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print(cm)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 13 - Gradient Boosting
    print('Gradient boosting:')
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.75, max_features=9, max_depth=3, random_state=0)
    y_pred = gb_clf.fit(X_train, y_train).predict(X_test)

    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

## classifier 14 - Neural Network
    print('Neural Network:')
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=3000, random_state=2)
    y_pred = mlp.fit(X_train, y_train).predict(X_test)
    print('Accuracy: ' + str(accuracy_score(y_test, y_pred)) + '\n')

################################


    # Plotting the distributions
    for var_name in ['tempo', 'beats', 'chroma_stft', 'rmse',
        'spectral_centroid', 'spectral_bandwidth', 'rolloff',
        'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
        'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
        'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
        'mfcc20']:
        distribution_plot(data[var_name], var_name)


    ###### distance matrix
    
    dist_mat = squareform(pdist(data))
    print(dist_mat)
        
    N = len(data)
    plt.pcolormesh(dist_mat)
    plt.colorbar()
    plt.xlim([0,N])
    plt.ylim([0,N])
    plt.show()
    ####################

    print(data.columns)
    print(data.cov())
    print(data.describe())
    print(data.corr())

    ## PCA
    projected = pca(x,y,2)
    label = [x//100 for x in range(1000)]
    colors = ['red','green','blue','purple','gray','brown','orange','cyan','pink','olive']
    plt.scatter(projected[:, 0], projected[:, 1],
            c=label, edgecolor='none', alpha=0.5,
            cmap=pltclrs.ListedColormap(colors))
    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.show()

if __name__ == "__main__":
    main()