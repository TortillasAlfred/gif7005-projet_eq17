from scorers.coveo_scorer import coveo_score

from scipy.spatial.distance import cdist

from joblib import Parallel, delayed, dump

import numpy as np

class CosineSimilarityRegressor(object):
    def __init__(self, all_docs, n_vectors_query, n_vectors_docs):
        self.all_docs = all_docs
        self.n_vectors_query = n_vectors_query
        self.n_vectors_docs = n_vectors_docs

    def fit(self, X, y):
        # Rien à faire, c'est un clf basé sur le prior
        pass

    def compute_cosine_similarities(self, x_i):
        distances = np.empty(shape=(self.all_docs.shape[0], ), dtype=object)

        for i, docs_i in enumerate(self.all_docs):
            dist_i = (1 - cdist(x_i, docs_i, metric="cosine")).ravel()
            dist_i = dist_i[~np.isnan(dist_i)]

            if dist_i.shape[0] == 0:
                dist_i = np.asarray([-1.0])
            
            distances[i] = dist_i

        return distances

    def compute_similarities_docs_included(self, X):
        queries = X[:, :self.n_vectors_query]
        docs = X[:, self.n_vectors_query:]

        distances = np.asarray([cdist(q, d, metric="cosine") for q, d in zip(queries, docs)])
        distances = distances.reshape(distances.shape[0], distances.shape[1] * distances.shape[2])

        similarities = 1.0 - distances

        return np.ma.masked_invalid(similarities)

    def predict(self, X):
        raise NotImplementedError()

    def score(self, X, y_true):
        y_pred = self.predict(X)

        return coveo_score(y_true, y_pred)


class MaximumCosineSimilarityRegressor(CosineSimilarityRegressor):
    def __init__(self, all_docs, n_vectors_query, n_vectors_docs):
        super().__init__(all_docs, n_vectors_query, n_vectors_docs)

    def predict(self, X, n_predicted_per_sample=5):
        y_predict = np.asarray(Parallel(n_jobs=-1, verbose=1)(delayed(self.predict_x)
                                    (x_i, n_predicted_per_sample) for x_i in X))
        
        if n_predicted_per_sample == -1:
            return y_predict
        else:
            return np.argpartition(y_predict, -n_predicted_per_sample)[:, -n_predicted_per_sample:]
            
    def predict_x(self, x_i, n_predicted_per_sample=5):
        distances = self.compute_cosine_similarities(x_i)

        return np.asarray([d.max() for d in distances])

class MeanCosineSimilarityRegressor(CosineSimilarityRegressor):
    def __init__(self, all_docs, n_vectors_query, n_vectors_docs):
        super().__init__(all_docs, n_vectors_query, n_vectors_docs)

    def predict(self, X, n_predicted_per_sample=5):
        y_predict = np.asarray(Parallel(n_jobs=-1, verbose=1)(delayed(self.predict_x)
                                    (x_i, n_predicted_per_sample) for x_i in X))
        
        if n_predicted_per_sample == -1:
            return y_predict
        else:
            return np.argpartition(y_predict, -n_predicted_per_sample)[:, -n_predicted_per_sample:]
            
    def predict_x(self, x_i, n_predicted_per_sample=5):
        distances = self.compute_similarities_docs_included(x_i)

        return np.asarray([np.mean(d) for d in distances])

class MeanMaxCosineSimilarityRegressor(CosineSimilarityRegressor):
    def __init__(self, all_docs, n_vectors_query, n_vectors_docs):
        super().__init__(all_docs, n_vectors_query, n_vectors_docs)

    def predict(self, X, n_predicted_per_sample=5):
        distances = self.compute_similarities_docs_included(X)

        y_predict = np.asarray([(np.ma.mean(d) + np.ma.max(d))/2 for d in distances])
        
        if n_predicted_per_sample == -1:
            return y_predict
        else:
            return np.argpartition(y_predict, -n_predicted_per_sample)[:, -n_predicted_per_sample:]


class QueryDocCosineSimilarityRegressor(object):
    def __init__(self, clf, n_vectors_query, n_vectors_docs):
        self.clf = clf
        self.n_vectors_query = n_vectors_query
        self.n_vectors_docs = n_vectors_docs
        self.n_partial_fits = 2500

    def partial_fit(self, X, y, class_weights):
        self.clf.partial_fit(self.compute_sorted_similarities(X), y, class_weights)

        self.n_partial_fits += 1
        if self.n_partial_fits % 500 == 0:
            dump(self.clf, "./data/cosine_clf_LR_" + str(self.n_partial_fits) + ".pck")

    def compute_sorted_similarities(self, X):
        queries = X[:, :self.n_vectors_query]
        docs = X[:, self.n_vectors_query:]

        distances = np.asarray([cdist(q, d, metric="cosine") for q, d in zip(queries, docs)])
        distances = distances.reshape(distances.shape[0], distances.shape[1] * distances.shape[2])

        similarities = 1.0 - distances
        similarities[np.isnan(similarities)] = 0

        return np.sort(similarities)[:, ::-1]

    def predict(self, X):
        return self.clf.predict(self.compute_sorted_similarities(X))
