import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings

import data

def basic_model():
    model = RandomForestClassifier(n_estimators=100)
    labels, targets, features = data.get_all("features.csv")
    features = data.normalize(features)

    scores = cross_val_score(model, features, targets, cv=5)

    return scores

def general_model():
    model = RandomForestClassifier(n_estimators=100)
    labels, targets, features = data.get_all("features.csv")

    logo = LeaveOneGroupOut()
    scores = []
    for train_index, test_index in logo.split(features, targets, groups=labels):
        features_copy = np.copy(features)
        X_train, X_test = data.normalize_train_test(features_copy[train_index], features_copy[test_index])
        y_train, y_test = targets[train_index], targets[test_index]

        model.fit(X_train, y_train)

        user_score = model.score(X_test, y_test)
        scores.append(user_score)

    return np.array(scores)

def personalized_model():
    model = RandomForestClassifier(n_estimators=100)
    labels, targets, features = data.get_all("features.csv")

    logo = LeaveOneGroupOut()
    scores = []
    for rest, user in logo.split(features, targets, groups=labels):
        normalized = data.normalize(features[user])

        user_scores = cross_val_score(model, normalized, targets[user], cv=5)

        scores = scores + user_scores.tolist()

    return np.array(scores)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Warning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    basic_model_scores = basic_model()
    print("Basic Model Accuracy: %0.2f (+/- %0.2f)" % (basic_model_scores.mean(), basic_model_scores.std() * 2))

    general_model_scores = general_model()
    print("General Model Accuracy: %0.2f (+/- %0.2f)" % (general_model_scores.mean(), general_model_scores.std() * 2))

    personalized_model_scores = personalized_model()
    print("Personalized Model Accuracy: %0.2f (+/- %0.2f)" % (personalized_model_scores.mean(), personalized_model_scores.std() * 2))