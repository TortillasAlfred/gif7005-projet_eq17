from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.wordVectorizer import WordVectorizer
from loading.queryTitleDataLoader import QueryTitleDataLoader
from wrappers.regression_wrapper import RegressionWrapper
from scorers.coveo_scorer import coveo_score

from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import numpy as np

import time

class NLP:
    def __init__(self, load_from_numpy):
        vectWV = WordVectorizer()
        self.loader_wv = QueryTitleDataLoader(vectorizer=vectWV, data_folder_path="./data/",
                                              numpy_folder_path="D:/machine_learning/data/wv/", load_from_numpy=load_from_numpy,
                                              load_dummy=False)

    def run_experiment(self):
        self.run_wn()

    def run_wn(self):
        print("**** WORD VECTOR ****")
        X_train, X_valid, _, y_train, y_valid = self.loader_wv.load_transform_data()

        reg = LinearRegression()
        reg.fit(X_train, y_train)
        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))
