import numpy as np

def read_csv(filename):
    features = np.genfromtxt(filename, dtype=str,delimiter=",", deletechars='"')    
    return features[1:]

def get_all(filename):
    features = read_csv(filename)
    return split_features(features)

def split_features(features):
    labels = features.T[-1]
    targets = features.T[-2]

    labels = np.char.strip(labels, '"')

    unique = np.unique(targets)
    for i in range(len(unique)):
        targets[targets == unique[i]] = i
    targets = targets.astype(np.int8)
    
    features = features.T[0:-2].T
    features = features.astype(np.float32)

    return labels, targets, features

def normalize(features):
    features = features.T
    for i in range(len(features)):
        max = np.amax(features[i])
        min = np.amin(features[i])
        features[i] = (features[i] - min) / (max - min)
    return features.T

def normalize_train_test(train, test):
    train, test = train.T, test.T
    for i in range(len(train)):
        max = np.amax(train[i])
        min = np.amin(train[i])
        temp = test[i]
        temp[temp > max] = max
        temp[temp < min] = min
        train[i] = (train[i] - min) / (max - min)
        test[i] = (temp - min) / (max - min)
    return train.T, test.T

if __name__ == "__main__":
    labels, targets, features = get_all("features.csv")
    print(labels.shape, targets.shape, features.shape)