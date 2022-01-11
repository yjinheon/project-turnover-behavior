import numpy as np
from collections import Counter


def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    self __init__(self, k=3):
        self.k = k 

    def fit(self, X, y): # triain sample and label
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array


    def _predict(self,x):
        distances = [euclidean_distance(x,x_train) for x_train in X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]



def smote_imp(X,k):
    imputed = []
    for i in range(len(X)):
        dd= KNN()
        
        impute = X[i] + add
        imputed.append(impute)

    return imputed