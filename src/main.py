import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings

import data

def basic_model():
    """
    TODO:
        Initialize a classifier
        Suggested classifiers:
            * SVM
            * K-nn
            * Decision Tree
            * Random Forrest
        Train a model and evaluate it
            You should use k-fold cross validation
        Return the accuracy of each fold in the scores variable
    """
    model = None
    """
        labels: a list with a subject-label for each feature-vector
        targets: the class labels
        features: the features
    """
    labels, targets, features = data.get_all("features.csv")
    features = data.normalize(features)

    scores = None

    return scores

def general_model():
    """
    TODO:
        Initialize a classifier
        Train a model and evaluate it
            Here you will use leave-one-group-out cross validation to create a general model
            Remember to normalize the training data set and the test data set seperately.
            There is a function to do this in data.py
        Return the accuracy of each fold in the scores variable as a numpy-array
    """
    model = None
    labels, targets, features = data.get_all("features.csv")

    scores = None

    return np.array(scores)

def personalized_model():
    """
    TODO:
        Initialize a classifier
        Train a model and evaluate it
            Here you will train one model per subject in the labels-list
            Remember to normalize the features for each subject seperately.
            Evaluate each model using k-fold cross validation
        Return the accuracy of each fold for every user in the scores variable as a numpy-array
    """
    model = None
    labels, targets, features = data.get_all("features.csv")

    scores = None

    return np.array(scores)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Warning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    basic_model_scores = basic_model()
    if basic_model_scores != None:
        print("Basic Model Accuracy: %0.2f (+/- %0.2f)" % (basic_model_scores.mean(), basic_model_scores.std() * 2))
    else:
        print("Implement the basic model")

    general_model_scores = general_model()
    if general_model_scores != None:
        print("General Model Accuracy: %0.2f (+/- %0.2f)" % (general_model_scores.mean(), general_model_scores.std() * 2))
    else:
        print("Implement the general model")

    personalized_model_scores = personalized_model()
    if personalized_model_scores != None:
        print("Personalized Model Accuracy: %0.2f (+/- %0.2f)" % (personalized_model_scores.mean(), personalized_model_scores.std() * 2))
    else:
        print("Implement the personalized model")