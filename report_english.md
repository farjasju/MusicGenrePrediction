# Music Genre Classification

## Introduction

### What is it exactly?

![](img/intro.png)

This project has a simple aim: determine the genre of a song based only on its audio characteristics (no meta-data). These characteristics are for example tempo, 

This problem is a *classification* problem: for each song, we need to attribute a class (a genre). Classification problems are solved using *machine learning* approaches, which have proved to be quite successful in extracting trends and patterns from large datasets. More specifically, our problem is a *supervised* classification problem: the model will learn from labeled data (i.e. each example song has a determined genre).

The possible classes in our problem are: *blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae* & *rock*.

The whole project is and relies on open source code.

### What has been done before

- Tzanetakis, George & Cook, Perry. (2002). *Musical Genre Classification of Audio Signals*. IEEE Transactions on Speech and Audio Processing. 10. 293 - 302. 10.1109/TSA.2002.800560
- Using this very dataset, a study was made on this [github repository](https://github.com/Insiyaa/Music-Tagging), achieving a 63% accuracy.
- study 2
- study 3

## Dataset & tools

### The data

The data is available on the *Kaggle* dataset [Music Features](https://www.kaggle.com/insiyeah/musicfeatures). It has been built from 1000 30-second audio tracks of 10 different genres, 100 tracks per genre. Each track is a 22050Hz Mono 16-bit audio file in .wav format. The features present in the dataset have been extracted from the songs using [libROSA](https://librosa.github.io/librosa/) library.

### The tools

The tools used in this project are Python and *scikit-learn* along with *pandas* and *numpy* for data analysis, and *matplotlib* for data visualization. 

## Methodology

### How we tried to solve this problem

#### Pre-processing

The main processing of the raw (audio) data has been done upstream, as the dataset already contains the features wanted. 

The processing of the 

#### Data exploration

- LDA

  

- Confusion matrix

  

#### Comparison of models

- List of models: linear/non-linear

  

- Cross-validation

  

#### Adjust of parameters



### The mathematical theory behind the models used



## Results

### Visualization and characterization of the data



### Results of the linear models



### Results of the non-linear models



### Comparison and discussion of the results



## Conclusions

### Discussion on the characteristics of the problem

- few training examples
- classifications of genres are often arbitrary and controversial

### What's the best model?



### Possible improvements

- subgenres (hierarchical structure)