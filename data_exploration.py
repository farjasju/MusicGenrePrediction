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
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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

def plot_confusion_matrix(matrix, show=True, save=False, filename=None, title='', subtitle=''):
    labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop',
 'reggae', 'rock']
    ax = sns.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels)
    ax.set_ylim(10.0, 0)
    plt.suptitle(title, fontsize=9)
    plt.title(subtitle, {'fontsize': 9})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    ax.set_yticklabels(ax.get_yticklabels(), rotation =0)
    ax.set_xticklabels(ax.get_yticklabels(), rotation =90)
    plt.margins(0.2)

    # ax.set_xticklabels(['',''] + labels, {'fontsize': 8,
    #     'verticalalignment': 'baseline'})
    # ax.set_yticklabels(['',''] + labels, {'fontsize': 8,
    #     'verticalalignment': 'baseline'})
    plt.subplots_adjust(bottom=0.15)
    # plt.xticks(rotation=90)
    if show:
        plt.show()
    if save:
        if not filename:
            raise Exception('Please specify a filename for the plot image')
        else:
            fig = ax.get_figure()
            fig.savefig(os.path.join('results',filename + '_cm.png'))
    plt.clf()

def run_classifier(classifier, X_train, X_test, y_train, y_test):
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return classifier, y_pred, cm

def title_generator(items):
    for i in range(len(items)):
        items[i] = str(items[i])
    return ' · '.join(items)

def subtitle_generator(y_test, y_pred):
    acc = str(round(accuracy_score(y_test, y_pred),2))
    prec = str(round(metrics.precision_score(y_test, y_pred, average='macro'),2))
    rec = str(round(metrics.recall_score(y_test, y_pred, average='macro'),2))
    f1 = str(round(metrics.f1_score(y_test, y_pred, average='macro'),2))
    # print("ROC AUC score:", str(metrics.roc_auc_score(y_pred, y_test, average='macro')))
    subtitle = ["acc=" + acc, "prec=" + prec, "rec=" + rec, "f1=" + f1]
    return title_generator(subtitle)

def cross_validation(clf, X, y):
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    return scores.mean()

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

    X_scaled = sc.fit_transform(X)
    X_lda = lda.fit_transform(X_scaled, y)

    # scatter_plot(pd.DataFrame(X_train)) ##scatterplot to see the LDA
    
    classifiers = []

    ## classifier 1 - random forest

    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    title = title_generator(["Random Forest", "100 estimators"])
    classifiers.append({'title': title, 'model':classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='random_forest', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)
    

    ## classifier 2 - naive bayes

    classifier = GaussianNB()
    title = title_generator(["Naive Bayes"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='naive_bayes', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 3 - logistic regression

    classifier = LogisticRegression(solver='sag', multi_class='auto')
    title = title_generator(["Logistic Regression", "Solver:'sag'"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='logistic_regression', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 4 - linear SVM

    classifier = SVC(kernel='linear')
    title = title_generator(["Linear SVM"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='linear_svm', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 5 - SVC poly kernel degree 2

    classifier = SVC(kernel='poly', degree=2, gamma="scale")
    title = title_generator(["SVC", "kernel='poly'", "degree=2", "gamma='scale'"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='svc_poly_2', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 6 - SVC poly kernel degree 4

    classifier = SVC(kernel='poly', degree=4, gamma="scale")
    title = title_generator(["SVC", "kernel='poly'", "degree=4", "gamma='scale'"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='svc_poly_4', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 7 - SVC gaussian kernel (best so far)

    classifier = SVC(kernel='rbf', gamma="scale", random_state=0)
    title = title_generator(["SVC", "kernel='rbf'", "gamma='scale'", "C=1"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='svc_rbf', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 8 - SVC sigmoid kernel

    classifier = SVC(kernel='sigmoid', gamma="scale")
    title = title_generator(["SVC", "kernel='sigmoid'", "gamma='scale'"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='svc_sigmoid', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 9 - Decision tree

    classifier = tree.DecisionTreeClassifier()
    title = title_generator(["Decision Tree"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='decision_tree', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 10 - KNN 5 neighbors

    classifier = KNeighborsClassifier(n_neighbors=5)
    title = title_generator(["Knn", "5 neighbors"])
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='knn_5', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 11 - KNN 7 neighbors

    classifier = KNeighborsClassifier(n_neighbors=7)
    title = title_generator(["Knn", "7 neighbors"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='knn_7', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 12 - KNN 9 neighbors

    classifier = KNeighborsClassifier(n_neighbors=9)
    title = title_generator(["Knn", "9 neighbors"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='knn_9', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 13 - Gradient Boosting

    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.75, max_features=9, max_depth=3, random_state=0)
    title = title_generator(["Gradient boosting", "n_estimators=100", "learning_rate=0.75", "max_features=9", "max_depth=3"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='gradient_boosting', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)

    ## classifier 14 - Neural Network

    classifier = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=3000, random_state=2)
    title = title_generator(["MLP", "hidden_layer_sizes=(8,8,8)", "activation='relu'", "solver='adam'", "max_iter=3000"])
    classifiers.append({'title': title, 'model': classifier})
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='mlp', title=title, subtitle=subtitle)
    # print(cross_validation(classifier, X_lda, y), title)
    
    for classifier in classifiers:
        print(cross_validation(classifier['model'], X_lda, y), classifier['title'])

    ################################

    # Tweaking SVM parameters
    # list_of_gammas = np.linspace(0.000001,1, 50)
    # list_of_C = [x / 10 for x in range(1, 20, 2)]
    # list_of_f1 = []
    # df_dict = {}
    # for gamma in list_of_gammas:
    #     for C in list_of_C:
    #         classifier = SVC(kernel='rbf', gamma=gamma, random_state=0, C=C)
    #         title = title_generator(["SVC", "kernel='rbf'", "gamma="+str(gamma), "C="+str(C)])
    #         model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    #         subtitle = subtitle_generator(y_test, y_pred)
    #         f1 = round(metrics.f1_score(y_test, y_pred, average='macro'),2)
    #         df_dict.setdefault('gamma', []).append(gamma)
    #         df_dict.setdefault('f1', []).append(f1)
    #         df_dict.setdefault('C', []).append(C)
    #         # print(title)
    #         # print(subtitle)
    # print(df_dict)
    # plt.title("SVM kernel='rbf', gamma='scale'")
    # plot_data = pd.DataFrame(df_dict)
    # print(plot_data)
    # sns.lineplot(x="gamma", y="f1", hue="C", data=plot_data, legend="full")
    # plt.show()

    # classifier = SVC(kernel='rbf', gamma=0.22, random_state=0, C=1.9)
    # title = title_generator(["SVC", "kernel='rbf'", "gamma=0.25", "C=1.5"])
    # model, y_pred, cm = run_classifier(classifier, X_train, X_test, y_train, y_test)
    # subtitle = subtitle_generator(y_test, y_pred)
    # plot_confusion_matrix(cm, show=False, save=True, filename='svc_rbf_tweaked', title=title, subtitle=subtitle)

    # print(cross_validation(classifier, X_lda, y))
    # print(cross_validation(classifier, X_scaled, y))

        # plot_confusion_matrix(cm, show=False, save=True, filename='svc_rbf_C'+str(C), title=title, subtitle=subtitle)

    # Plotting the distributions
    # for var_name in ['tempo', 'beats', 'chroma_stft', 'rmse',
    #     'spectral_centroid', 'spectral_bandwidth', 'rolloff',
    #     'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
    #     'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
    #     'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19',
    #     'mfcc20']:
    #     distribution_plot(data[var_name], var_name)


    ###### distance matrix
    
    # dist_mat = squareform(pdist(data))
    # print(dist_mat)
        
    # N = len(data)
    # plt.pcolormesh(dist_mat)
    # plt.colorbar()
    # plt.xlim([0,N])
    # plt.ylim([0,N])
    # plt.show()
    ####################

    # print(data.columns)
    # print(data.cov())
    # print(data.describe())
    # print(data.corr())

    ## PCA
    # projected = pca(x,y,2)
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
