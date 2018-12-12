from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.wordVectorizer import WordVectorizer

from loading.dataLoader import DataLoader
from wrappers.regression_wrapper import RegressionWrapper, MultiOutputRegressorWrapper
from scorers.coveo_scorer import coveo_score

from sklearn.linear_model import LogisticRegression, LinearRegression
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
                                    load_from_numpy=load_from_numpy, filter_no_clicks=True)
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
        # self.run_wn()
        # self.run_unfiltered()
        # self.run_filtered()
        self.run_multioutput_lin_reg_balanced_wv()
        # self.run_multioutput_lin_reg_balanced_BoW()

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
    

    def run_multioutput_lin_reg_balanced_wv(self):
        # ROule en 90 min sur mon PC
        print("**** MULTIOUTPUT LR BALANCED WV ****")
        self.loader_wv.load_transform_data()
        X_train, y_train, all_docs = self.loader_wv.load_all_from_numpy("X_train", "y_train", "all_docs")

        reg = MultiOutputRegressorWrapper(LinearRegression(), n_jobs=-1, total_outputs=all_docs.shape[0])
        del all_docs
        print("BEGIN FIT")
        reg.fit(X_train, y_train)
        print("BEGIN PREDICT TRAIN")
        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        del X_train
        del y_train
        X_valid, y_valid = self.loader_wv.load_all_from_numpy("X_valid", "y_valid")
        print("BEGIN PREDICT VALID")
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))

    def run_multioutput_lin_reg_balanced_BoW(self):
        print("**** MULTIOUTPUT LR BALANCED BoW ****")
        self.loader_unfiltered.load_transform_data()
        X_train, y_train, all_docs = self.loader_filtered.load_all_from_numpy("X_train", "y_train", "all_docs")

        reg = MultiOutputRegressorWrapper(LinearRegression(), n_jobs=-1, total_outputs=all_docs.shape[0])
        del all_docs
        print("BEGIN FIT")
        reg.fit(X_train, y_train)
        print("BEGIN PREDICT TRAIN")
        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        del X_train
        del y_train
        X_valid, y_valid = self.loader_wv.load_all_from_numpy("X_valid", "y_valid")
        print("BEGIN PREDICT VALID")
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))