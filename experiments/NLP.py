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
        vectBOW = BagOfWordsVectorizer()
        self.loader_wv = QueryTitleDataLoader(vectorizer=vectWV, data_folder_path="./data/",
                                              numpy_folder_path="./data/wv/", load_from_numpy=load_from_numpy,
                                              load_dummy=True)
        self.loader_unfiltered = QueryTitleDataLoader(vectorizer=vectBOW, data_folder_path="./data/",
                                                      numpy_folder_path="./data/bow_oh_unfiltered/", load_from_numpy=load_from_numpy,
                                                      load_dummy=True)
        self.loader_filtered = QueryTitleDataLoader(vectorizer=vectBOW, data_folder_path="./data/",
                                                    numpy_folder_path="./data/bow_oh_filtered/", load_from_numpy=load_from_numpy,
                                                    load_dummy=True)

    def run_experiment(self):
        self.run_wn()
        self.run_unfiltered()
        self.run_filtered()

    def run_wn(self):
        print("**** WORD VECTOR ****")
        X_train, X_valid, _, y_train, y_valid = self.loader_wv.load_transform_data()

        reg = LinearRegression()
        reg.fit(X_train, y_train)
        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))

    def run_unfiltered(self):
        print("**** UNFILTERED EXP ****")
        X_train, X_valid, _, y_train, y_valid, _, all_docs_ids = self.loader_unfiltered.load_transform_data()

        reg = RegressionWrapper(LinearRegression(), total_outputs=all_docs_ids.shape[0])
        reg.fit(X_train, y_train)
        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))

    def run_filtered(self):
        print("**** FILTERED EXP ****")
        X_train, X_valid, _, y_train, y_valid, _, all_docs_ids = self.loader_filtered.load_transform_data()

        reg = RegressionWrapper(LinearRegression(), total_outputs=all_docs_ids.shape[0])
        reg.fit(X_train, y_train)
        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))
