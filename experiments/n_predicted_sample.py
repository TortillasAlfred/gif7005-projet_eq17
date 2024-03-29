from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.wordVectorizer import MatrixWordVectorizer
from loading.dataLoader import DataLoader
from wrappers.regression_wrapper import RegressionWrapper, MultiOutputRegressorWrapper
from wrappers.qd_regression_wrapper import QueryDocRegressionWrapper
from learners.cosine_similarity import *

from scorers.coveo_scorer import coveo_score

import seaborn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.base import clone

from joblib import Parallel, delayed, load

import numpy as np

class GlobalExperiment():
    def __init__(self, load_from_numpy=False):
        vectWV = MatrixWordVectorizer()
        vectBOW = BagOfWordsVectorizer()
        enc = OneHotEncoder()
        self.loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.only_query,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/qd_wv_matrix/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=True)
        self.loader_wv.load_transform_data()
        self.loader_filtered = DataLoader(vectorizer=vectBOW, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/bow_oh_filtered/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=True)

    def run_experiment_Cosine_LR(self):
        # self.collect_results_cosine_LR()
        self.plot_results_cosine_LR()

    def plot_results_cosine_LR(self):        
        for n_fits in [200, 400, 600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]:
            plt.plot(np.load("./data/mean_max_train_scores_" + str(n_fits) + ".npy"), label="Train {} fits".format(n_fits))
            plt.plot(np.load("./data/mean_max_valid_scores_" + str(n_fits) + ".npy"), label="Valid {} fits".format(n_fits))

        plt.xlabel("N_predicted_per_sample")
        plt.ylabel("Coveo score")
        plt.title("Métriques avec un LR entraîné sur la cosine similarity")

        plt.legend()

        plt.show()
        

    def collect_results_cosine_LR(self):
        X_train, X_valid, y_train, y_valid, all_docs = self.loader_wv.load_all_from_numpy("X_train", "X_valid",
                                                                                        "y_train", "y_valid", "all_docs")

        for n_fits in [1500, 2000, 2500, 3000, 3500, 4000, 4500]:
            print(n_fits)
            reg = load("./data/cosine_clf_LR_" + str(n_fits) + ".pck")
            clf = QueryDocRegressionWrapper(QueryDocCosineSimilarityRegressor(reg, 20, 20),
                                            all_docs, proportion_neg_examples=-1,
                                            matrix_embeddings=True, n_jobs=6)
            scores_train = self.get_all_scores(clf.predict(X_train, n_predicted_per_sample=-1), y_train)
            scores_valid = self.get_all_scores(clf.predict(X_valid, n_predicted_per_sample=-1), y_valid)
            
            np.save("./data/mean_max_train_scores_" + str(n_fits), scores_train)
            np.save("./data/mean_max_valid_scores_" + str(n_fits), scores_valid)


    def run_experiment_LR(self):
        self.run_experiment(LinearRegression(), "Linear Regression")

    def run_experiment_Cosine(self):
        self.collect_results_cosine()
        self.plot_results_cosine()

    def plot_results_cosine(self):
        s_t_max = np.load("./data/max_train_scores.npy")
        s_v_max = np.load("./data/max_valid_scores.npy")
        s_t_mean = np.load("./data/mean_train_scores.npy")
        s_v_mean = np.load("./data/mean_valid_scores.npy")
        s_t_mean_max = np.load("./data/mean_max_train_scores.npy")
        s_v_mean_max = np.load("./data/mean_max_valid_scores.npy")

        plt.plot(s_t_max, label="Train_Max")
        plt.plot(s_v_max, label="Valid_Max")
        plt.plot(s_t_mean, label="Train_Mean")
        plt.plot(s_v_mean, label="Valid_Mean")
        plt.plot(s_t_mean_max, label="Train_Mean_Max")
        plt.plot(s_v_mean_max, label="Valid_Mean_Max")
        
        plt.xlabel("N_predicted_per_sample")
        plt.ylabel("Coveo score")
        plt.title("Métriques des différents clf de Cosine")

        plt.legend()

        plt.show()
        

    def collect_results_cosine(self):
        X_train, X_valid, y_train, y_valid, all_docs = self.loader_wv.load_all_from_numpy("X_train", "X_valid",
                                                                                        "y_train", "y_valid", "all_docs")

        print("*** MAX ***")                              
        clf = MaximumCosineSimilarityRegressor(all_docs, 20, 20)
        scores_train = self.get_all_scores(clf.predict(X_train, n_predicted_per_sample=-1), y_train)
        scores_valid = self.get_all_scores(clf.predict(X_valid, n_predicted_per_sample=-1), y_valid)
        
        np.save("./data/max_train_scores", scores_train)
        np.save("./data/max_valid_scores", scores_valid)

        print("*** MEAN ***")
        clf = MeanCosineSimilarityRegressor(all_docs, 20, 20)
        scores_train = self.get_all_scores(clf.predict(X_train, n_predicted_per_sample=-1), y_train)
        scores_valid = self.get_all_scores(clf.predict(X_valid, n_predicted_per_sample=-1), y_valid)
        
        np.save("./data/mean_train_scores", scores_train)
        np.save("./data/mean_valid_scores", scores_valid)

        print("*** MEAN MAX ***")
        clf = MeanMaxCosineSimilarityRegressor(all_docs, 20, 20)
        scores_train = self.get_all_scores(clf.predict(X_train, n_predicted_per_sample=-1), y_train)
        scores_valid = self.get_all_scores(clf.predict(X_valid, n_predicted_per_sample=-1), y_valid)
        
        np.save("./data/mean_max_train_scores", scores_train)
        np.save("./data/mean_max_valid_scores", scores_valid)

    def run_experiment_LR_balanced(self):
        self.run_experiment_mop_balanced(LinearRegression(), "Linear Regression balanced")


    def run_experiment_mop_balanced(self, clf, clf_name):
        self.transform_data()
        scores_train_we, scores_valid_we = self.collect_results_mop_balanced(self.loader_wv, clone(clf))
        # scores_train_bow, scores_valid_bow = self.collect_results_mop_balanced(self.loader_filtered, clone(clf))
        plt.plot(scores_train_we, label="Train_We")
        plt.plot(scores_valid_we, label="Valid_We")
        
        plt.xlabel("N_predicted_per_sample")
        plt.ylabel("Coveo score")
        plt.title(clf_name)

        plt.legend()

        plt.show()


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

    def collect_results_mop_balanced(self, loader, clf):
        X_train, y_train, all_docs_ids = loader.load_all_from_numpy("X_train", "y_train", "all_docs_ids")

        reg = MultiOutputRegressorWrapper(clf, total_outputs=all_docs_ids.shape[0])
        del all_docs_ids
        reg.fit(X_train, y_train)
        scores_train = self.get_all_scores(reg.predict(X_train, n_predicted_per_sample=-1), y_train)
        del X_train, y_train
        X_valid, y_valid = loader.load_all_from_numpy("X_valid", "y_valid")
        scores_valid = self.get_all_scores(reg.predict(X_valid, n_predicted_per_sample=-1), y_valid)

        return scores_train, scores_valid

    
    def get_all_scores(self, y_predict, y_true):
        y_predict = np.argsort(y_predict)[:, ::-1]

        all_scores = Parallel(n_jobs=1, verbose=10)(delayed(coveo_score)(y_true, y_predict[:, :n]) for n in range(1, 500))

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