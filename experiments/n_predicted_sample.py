from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.wordVectorizer import WordVectorizer
from loading.dataLoader import DataLoader
from wrappers.regression_wrapper import RegressionWrapper

from scorers.coveo_scorer import coveo_score

import seaborn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.base import clone

from joblib import Parallel, delayed

import numpy as np

class GlobalExperiment():
    def __init__(self, load_from_numpy=False):
        vectWV = WordVectorizer()
        vectBOW = BagOfWordsVectorizer()
        enc = OneHotEncoder()
        self.loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/wv/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=True)
        self.loader_filtered = DataLoader(vectorizer=vectBOW, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/bow_oh_filtered/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=True)


    def run_experiment_LR(self):
        self.run_experiment(LinearRegression(), "Linear Regression")


    def run_experiment(self, clf, clf_name):
        self.transform_data()
        scores_train_we, scores_valid_we = self.collect_results(self.loader_wv, clone(clf))
        scores_train_bow, scores_valid_bow = self.collect_results(self.loader_filtered, clone(clf))
        self.plot_scores(scores_train_we, scores_valid_we, scores_train_bow, scores_valid_bow, clf_name)

    
    def transform_data(self):
        self.loader_filtered.load_transform_data()
        self.loader_wv.load_transform_data()

    
    def collect_results(self, loader, clf):
        X_train, X_valid, y_train, y_valid, all_docs_ids = loader.load_all_from_numpy("X_train", "X_valid",
                                                                                        "y_train", "y_valid", "all_docs_ids")

        reg = RegressionWrapper(clf, total_outputs=all_docs_ids.shape[0])
        reg.fit(X_train, y_train)
        scores_train = self.get_all_scores(reg.predict(X_train, n_predicted_per_sample=-1), y_train)
        scores_valid = self.get_all_scores(reg.predict(X_valid, n_predicted_per_sample=-1), y_valid)

        return scores_train, scores_valid

    
    def get_all_scores(self, y_predict, y_true):
        y_predict = np.argsort(y_predict)[:, ::-1]

        all_scores = Parallel(n_jobs=-1, verbose=10)(delayed(coveo_score)(y_true, y_predict[:, :n]) for n in range(1, 1000))

        return all_scores


    def plot_scores(self, scores_train_we, scores_valid_we, scores_train_bow, scores_valid_bow, clf_name):
        plt.plot(scores_train_bow, label="Train_BoW")
        plt.plot(scores_valid_bow, label="Valid_BoW")
        plt.plot(scores_train_we, label="Train_We")
        plt.plot(scores_valid_we, label="Valid_We")
        
        plt.xlabel("N_predicted_per_sample")
        plt.ylabel("Coveo score")
        plt.title(clf_name)

        plt.legend()

        plt.show()