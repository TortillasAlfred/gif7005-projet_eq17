import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scorers.coveo_scorer import coveo_score


class KNNWrapper:
    def __init__(self, n_neighbors, weights, algorithm):
        self.clf = KNeighborsClassifier(n_neighbors, weights, algorithm)

    def fit(self, X, y):
        self.X, self.y = X, y
        self.clf.fit(X, y)

    def predict(self, X, n_predicted_per_sample=5):
        y_predict = self.clf.predict(X)
        k_neighbors = self.clf.kneighbors(X, n_predicted_per_sample-1, return_distance=False)
        y_predict_raw = np.zeros((k_neighbors.shape[0], k_neighbors.shape[1] + 1))

        for i in range(len(y_predict)):
            y_predict_raw[i][0] = y_predict[i]
            for j in range(n_predicted_per_sample-1):
                y_predict_raw[i][j + 1] = self.y[k_neighbors[i, j]]

        return y_predict_raw

    def score(self, X, y_true):
        y_pred = self.predict(X)
        y_true = [[i] for i in y_true]
        return coveo_score(y_true, y_pred)