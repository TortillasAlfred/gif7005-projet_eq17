import numpy as np

from scorers.coveo_scorer import coveo_score

from sklearn.utils import shuffle

from random import choice

from joblib import Parallel, delayed

from math import ceil


class QueryDocRegressionWrapper:
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
        self.docs = np.asarray(docs, dtype="float16")
        self.proportion_neg_examples = proportion_neg_examples
        self.n_jobs = n_jobs
        self.n_predicted_per_sample = n_predicted_per_sample
        self.random_state = 42

    def partial_fit(self, X, y):
        if self.proportion_neg_examples != -1:
            raise AssertionError("partial_fit was made to be used when proportion_neg_examples is -1")
        print("BEGIN PARTIAL FIT")
        self.clf.partial_fit(X, y)

    def fit(self, X, y):
        print("BEGIN FIT")
        X_reg, y_reg = self.create_combinations(X, y)
        self.clf.fit(X_reg, y_reg)

    def create_combinations(self, X, y):
        n_docs = self.docs.shape[0]

        n_pos = np.sum([y_i.shape[0] for y_i in y])
        n_neg = n_pos * self.proportion_neg_examples
        X_reg = np.memmap("./data/X_reg.npy", mode="w+", shape=(n_pos + n_neg, X.shape[1] + self.docs.shape[1]), dtype="float16")
        X_reg.fill(0.0)
        y_reg = np.memmap("./data/y_reg.npy", mode="w+", shape=(n_pos + n_neg, ), dtype=bool)
        y_reg.fill(False)

        X_taken = set()
        docs_taken = set()
        next_idx = 0

        for i, y_i in enumerate(y):
            for doc_number in y_i:
                if next_idx % 5000 == 0:
                    print(next_idx)
                X_taken.add(i)
                docs_taken.add(doc_number)
                X_reg[next_idx] = np.hstack((X[i], self.docs[doc_number]))
                y_reg[next_idx] = True
                next_idx += 1

        X_left = [x_i for x_i in range(X.shape[0]) if x_i not in X_taken]
        docs_left = [d_i for d_i in range(n_docs) if d_i not in docs_taken]

        for _ in range(n_neg):
            if next_idx % 5000 == 0:
                print(next_idx)
            if len(X_left) == 0:
                X_left = list(range(X.shape[0]))
            if len(docs_left) == 0:
                docs_left = list(range(n_docs))
            while True:
                x_drawn = choice(X_left)
                doc_drawn = choice(docs_left)
                if doc_drawn not in y[x_drawn]:
                    break
            X_left.remove(x_drawn)
            docs_left.remove(doc_drawn)
            X_reg[next_idx] = np.hstack((X[x_drawn], self.docs[doc_drawn]))
            y_reg[next_idx] = False
            next_idx += 1

        return shuffle(X_reg, y_reg, random_state=self.random_state)

    def predict(self, X):
        print("BEGIN PREDICT")

        y_predict = Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(self.predict_x_i)(x_i) for x_i in X)

        return np.asarray(y_predict)

    def predict_x_i(self, x_i):
        y_i_raw = self.clf.predict([np.hstack((x_i, d)) for d in self.docs])

        return np.argpartition(y_i_raw, -self.n_predicted_per_sample)[-self.n_predicted_per_sample:]

    def score(self, X, y_true):
        y_pred = self.predict(X)

        return coveo_score(y_true, y_pred)
