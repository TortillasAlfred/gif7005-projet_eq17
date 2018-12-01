import numpy as np

from scorers.coveo_scorer import coveo_score

from random import choice

from joblib import Parallel, delayed


class RegressionWrapper:
    '''
    Wrapper pour les classificateurs par régression. Va principalement juste overloader le score pour avoir le score
    selon la métrique de Coveo et va aussi ajuster les y pour un output de regression.

    À ma connaissance, notre problème lorsque vu comme régression
    et du multioutput regression, parce qu'on demande de prédire un vecteur de taille 6000 environ. Et selon sklearn,
    le seul régresseur MultiOutput qui marche out of the box c'est LinearRegression, les autres sont single output.

    Alors, ça sera probablement nécessaire pour utiliser ce wrapper là de passer soit un LinearRegression ou encore
    un MultiOutput() avec n'importe quel autre régresseur comme clf. MultiOutput() va juste entraîner un classificateur
    pour chacune des 6000 valeurs de sortie, alors c'est horriblement lent. Je pense pas que ce sera vraiment utile.

    Le RegressionWrapper sera aussi utilisé par un éventuel réseau de neurones.
    '''
    def __init__(self, clf, docs, proportion_neg_examples, n_jobs=-1, n_predicted_per_sample=5):
        self.clf = clf
        self.docs = docs
        self.proportion_neg_examples = proportion_neg_examples
        self.n_jobs = n_jobs
        self.n_predicted_per_sample = n_predicted_per_sample

    def fit(self, X, y):
        X_reg, y_reg = self.create_combinations(X, y)
        self.clf.fit(X_reg, y_reg)

    def create_combinations(self, X, y):
        n_docs = self.docs.shape[0]

        n_pos = np.sum([y_i.shape[0] for y_i in y])
        n_neg = n_pos * self.proportion_neg_examples
        X_reg = np.zeros((n_pos + n_neg, X.shape[1] + self.docs.shape[1]), dtype="float32")
        y_reg = np.zeros((n_pos + n_neg, ), dtype=bool)

        X_taken = set()
        docs_taken = set()
        next_idx = 0

        for i, y_i in enumerate(y):
            X_taken.add(i)
            for doc_number in y_i:
                docs_taken.add(doc_number)
                X_reg[next_idx] = X[i]
                y_reg[next_idx] = True
                next_idx += 1

        X_left = [x_i for x_i in range(X.shape[0]) if x_i not in X_taken]
        docs_left = [d_i for d_i in range(n_docs) if d_i not in docs_taken]

        for _ in range(n_neg):
            while True:
                x_drawn = choice(X_left)
                doc_drawn = choice(docs_left)
                if doc_drawn not in y[x_drawn]:
                    break
            X_left.remove(x_drawn)
            docs_left.remove(doc_drawn)
            X_reg[next_idx] = X[x_drawn]
            y_reg[next_idx] = False
            next_idx += 1
            if len(X_left) == 0:
                X_left = range(X.shape[0])
            if len(docs_left) == 0:
                docs_left = range(n_docs)

        return X_reg, y_reg

    def predict(self, X):
        y_predict = []

        for x in X:
            y_i_raw = Parallel(n_jobs=self.n_jobs)(delayed(self.clf.predict)(np.hstack((x, d))) for d in self.docs)
            y_i = np.argpartition(y_i_raw, -self.n_predicted_per_sample)[:, -self.n_predicted_per_sample:]
            y_predict.append(y_i)

        return np.asarray(y_predict)

    def score(self, X, y_true):
        y_pred = self.predict(X)

        return coveo_score(y_true, y_pred)
