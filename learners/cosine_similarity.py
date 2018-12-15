from scorers.coveo_scorer import coveo_score

from scipy.spatial.distance import cdist

import numpy as np

class CosineSimilarityRegressor(object):
    def __init__(self, n_vectors_query, n_vectors_docs):
        self.n_vectors_query = n_vectors_query
        self.n_vectors_docs = n_vectors_docs

    def fit(self, X, y):
        # Rien à faire, c'est un clf basé sur le prior
        pass

    def compute_cosine_similarities(self, X):
        # Éliminer les zéros

        # Si une query ou un doc est vide (que des zéros), retourner 0
        pass

    def predict(self, X):
        raise NotImplementedError()

    def score(self, X, y_true):
        y_pred = self.predict(X)

        return coveo_score(y_true, y_pred)


class MaximumCosineSimilarityRegressor(CosineSimilarityRegressor):
    def __init__(self, n_vectors_query, n_vectors_docs):
        super().__init__(n_vectors_query, n_vectors_docs)

    def predict(self, X):
        dist_matrixes = self.compute_cosine_similarities(X)

        # Retourne le max
    