from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.wordVectorizer import WordVectorizer
from loading.dataLoader import DataLoader
from wrappers.regression_wrapper import RegressionWrapper
from scorers.coveo_scorer import coveo_score

from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import numpy as np

import time

class LR:
    def __init__(self, load_from_numpy):
        vectWV = WordVectorizer()
        vectBOW = BagOfWordsVectorizer()
        enc = OneHotEncoder()
        self.loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/wv/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=False)
        self.loader_unfiltered = DataLoader(vectorizer=vectBOW, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/bow_oh_unfiltered/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=False)
        self.loader_filtered = DataLoader(vectorizer=vectBOW, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/bow_oh_filtered/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=True)

    def run_experiment(self):
        self.run_wn()
        self.run_unfiltered()
        self.run_filtered()

    def run_wn(self):
        print("**** WORD VECTOR UNFILTERED ****")
        X_train, X_valid, _, y_train, y_valid, _, all_docs_ids = self.loader_wv.load_transform_data()

        reg = RegressionWrapper(LinearRegression(), total_outputs=all_docs_ids.shape[0])
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
