import numpy as np

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

from scorers.coveo_scorer import coveo_score

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

    def __init__(self, clf, total_outputs):
        self.clf = clf
        self.total_outputs = total_outputs

    def fit(self, X_train, y_train):
        y_train = self.make_regression_y(y_train)
        self.clf.fit(X_train, y_train)

    def score(self, X, y_true):
        y_pred = self.predict(X)

        return coveo_score(y_true, y_pred)

    def predict(self, X, n_predicted_per_sample=5):
        # Va retourner une matrice de taille [X.shape[0], n_predicted_per_sample]
        y_predict_raw = self.clf.predict(X)

        if n_predicted_per_sample == -1:
            return y_predict_raw
        else:
            return np.argpartition(y_predict_raw, -n_predicted_per_sample)[:, -n_predicted_per_sample:]


    def make_regression_y(self, y):
        new_y = np.zeros((y.shape[0], self.total_outputs), dtype=bool)

        for idx, y_i in enumerate(y):
            for y_i_j in y_i:
                new_y[idx, y_i_j] = 1

        return new_y

class MultiOutputRegressorWrapper(RegressionWrapper):
    def __init__(self, clf, total_outputs, n_jobs=-1):
        RegressionWrapper.__init__(self, clf=clf, total_outputs=total_outputs)
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        y_reg = self.make_regression_y(y)

        self.classifiers = Parallel(n_jobs=self.n_jobs, verbose=10)\
                                (delayed(self.train_model)(X, y_reg[:, i]) for i in range(self.total_outputs))

    def train_model(self, X, y):
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        clf_new = clone(self.clf)
        return clf_new.fit(X, y, sample_weight=sample_weights)

    def predict(self, X, n_predicted_per_sample=5):
        X_slices = np.array_split(X, int(X.shape[0]/1000))

        y_predict = np.asarray(Parallel(n_jobs=self.n_jobs, verbose=10)(delayed(self.predict_x)(x_slice, n_predicted_per_sample) for x_slice in X_slices))

        return np.vstack([y for y in y_predict])

    def predict_x(self, x, n_predicted_per_sample=5):
        y_predict = np.asarray([clf.predict(x) for clf in self.classifiers]).T

        return np.argpartition(y_predict, -n_predicted_per_sample)[:, -n_predicted_per_sample:]


    def score(self, X, y_true):
        y_pred = self.predict(X)

        return coveo_score(y_true, y_pred)
