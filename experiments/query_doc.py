from loading.wordVectorizer import DictSentenceVectorizerSpacy, MatrixWordVectorizer
from loading.dataLoader import DataLoader
from loading.queryDocBatchDataLoader import QueryDocBatchDataLoader
from loading.oneHotEncoder import OneHotEncoder

from learners.cosine_similarity import *
from learners.LSTM import CustomLSTM

from wrappers.qd_regression_wrapper import QueryDocRegressionWrapper
from wrappers.regression_wrapper import RegressionWrapper

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor

from joblib import load

class PoC:
    def __init__(self, load_from_numpy):
        vectWV = DictSentenceVectorizerSpacy()
        enc = OneHotEncoder()
        self.loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/qd_wv/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=True)
        self.batch_loader = QueryDocBatchDataLoader(vectorizer=vectWV, encoder=enc, batch_size=4e4, data_folder_path="./data/",
                                                    numpy_folder_path="./data/qd_wv/", load_from_numpy=load_from_numpy,
                                                    filter_no_clicks=True, load_dummy=False, generate_pairs=False)

    def run_experiment(self):
        # self.run_normal_wrapped()
        self.run_qd_wrapped_all_dataset()

    def run_qd_wrapped_all_dataset(self):
        print("**** QD-WRAPPED PAR REG ALL DATASET ****")
        all_docs = self.loader_wv.load_all_from_numpy("all_docs")

        reg = QueryDocRegressionWrapper(SGDRegressor(verbose=1), all_docs, class_weights=self.batch_loader.get_class_weights(), proportion_neg_examples=-1)

        while True:
            partial_X_train, partial_y_train = self.batch_loader.get_next_batch()
            if partial_X_train is None or partial_y_train is None:
                break
            reg.partial_fit(partial_X_train, partial_y_train)

        X_train, X_valid, y_train, y_valid = self.batch_loader.get_X_and_y()

        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))

    def run_normal_wrapped(self):
        print("**** NORMAL-WRAPPED LIN REG ****")
        self.loader_wv.load_transform_data()
        X_train, X_valid, y_train, y_valid, all_docs = self.loader_wv.load_all_from_numpy("X_train", "X_valid",
                                                                                          "y_train", "y_valid",
                                                                                          "all_docs")

        reg = RegressionWrapper(LinearRegression(), total_outputs=all_docs.shape[0])
        reg.fit(X_train, y_train)
        print("Coveo score on train : {}".format(reg.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(reg.score(X_valid, y_valid)))


class CosineClassifiers:
    def __init__(self, load_from_numpy):
        self.load_from_numpy = load_from_numpy

    def run_experiment(self):
        self.run_mean_max_cosine()
        self.run_mean_cosine()
        self.run_max_cosine()

    def run_max_cosine(self):
        vectWV = MatrixWordVectorizer()
        enc = OneHotEncoder()
        loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.only_query,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/qd_wv_matrix/",
                                    load_from_numpy=self.load_from_numpy, filter_no_clicks=True)
        loader_wv.load_transform_data()

        print("**** QD-WRAPPED MAXIMUM COSINE ****")
        all_docs = loader_wv.load_all_from_numpy("all_docs")

        X_train, X_valid, y_train, y_valid = loader_wv.load_all_from_numpy("X_train", "X_valid",
                                                                            "y_train", "y_valid")

        del loader_wv, vectWV, enc

        clf = MaximumCosineSimilarityRegressor(all_docs, X_train.shape[1], all_docs.shape[1])

        print("Coveo score on train : {}".format(clf.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(clf.score(X_valid, y_valid)))

    def run_mean_cosine(self):
        vectWV = MatrixWordVectorizer()
        enc = OneHotEncoder()
        loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.only_query,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/qd_wv_matrix/",
                                    load_from_numpy=self.load_from_numpy, filter_no_clicks=True)
        loader_wv.load_transform_data()

        print("**** QD-WRAPPED MEAN COSINE ****")
        all_docs = loader_wv.load_all_from_numpy("all_docs")

        X_train, X_valid, y_train, y_valid = loader_wv.load_all_from_numpy("X_train", "X_valid",
                                                                            "y_train", "y_valid")

        del loader_wv, vectWV, enc

        clf = MeanCosineSimilarityRegressor(all_docs, X_train.shape[1], all_docs.shape[1])

        print("Coveo score on train : {}".format(clf.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(clf.score(X_valid, y_valid)))

    def run_mean_max_cosine(self):
        vectWV = MatrixWordVectorizer()
        enc = OneHotEncoder()
        loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.only_query,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/qd_wv_matrix/",
                                    load_from_numpy=self.load_from_numpy, filter_no_clicks=True)
        loader_wv.load_transform_data()

        print("**** QD-WRAPPED MEAN MAX COSINE ****")
        all_docs = loader_wv.load_all_from_numpy("all_docs")

        X_train, X_valid, y_train, y_valid = loader_wv.load_all_from_numpy("X_train", "X_valid",
                                                                            "y_train", "y_valid")

        del loader_wv, vectWV, enc

        clf = MeanMaxCosineSimilarityRegressor(all_docs, X_train.shape[1], all_docs.shape[1])

        print("Coveo score on train : {}".format(clf.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(clf.score(X_valid, y_valid)))


class CosineSimilarityWithLinearRegressor:
    def __init__(self, load_from_numpy):
        vectWV = MatrixWordVectorizer()
        enc = OneHotEncoder()
        self.batch_loader = QueryDocBatchDataLoader(vectorizer=vectWV, encoder=enc, batch_size=4e4, data_folder_path="./data/",
                                                    numpy_folder_path="./data/qd_wv_matrix/", load_from_numpy=load_from_numpy,
                                                    filter_no_clicks=True, load_dummy=False, generate_pairs=False)

    def run_experiment(self):
        self.run_qd_wrapped_all_dataset()

    def run_qd_wrapped_all_dataset(self):
        print("**** Query Doc Cosine with LinReg ****")
        all_docs = self.batch_loader.load_all_from_numpy("all_docs")
        clf = QueryDocRegressionWrapper(QueryDocCosineSimilarityRegressor(SGDRegressor(verbose=1), 20, 20),
                                        docs=all_docs, proportion_neg_examples=-1, 
                                        class_weights=self.batch_loader.get_class_weights(),
                                        matrix_embeddings=True)

        n_epoch = 1
        max_epoch = 10
        while True:
            partial_X_train, partial_y_train = self.batch_loader.get_next_batch()
            if partial_X_train is None or partial_y_train is None:
                if n_epoch >= max_epoch:
                    break

                n_epoch += 1
                self.batch_loader.next_epoch()

            clf.partial_fit(partial_X_train, partial_y_train)

        X_train, X_valid, y_train, y_valid = self.batch_loader.get_X_and_y()

        print("Coveo score on train : {}".format(clf.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(clf.score(X_valid, y_valid)))


class LSTM_first_try:
    def __init__(self, load_from_numpy):
        vectWV = MatrixWordVectorizer()
        enc = OneHotEncoder()
        self.batch_loader = QueryDocBatchDataLoader(vectorizer=vectWV, encoder=enc, batch_size=4e4, data_folder_path="./data/",
                                                    numpy_folder_path="./data/qd_wv_matrix/", load_from_numpy=load_from_numpy,
                                                    filter_no_clicks=True, load_dummy=False, generate_pairs=False)

    def run_experiment(self):
        self.run_qd_wrapped_all_dataset()

    def run_qd_wrapped_all_dataset(self):
        print("**** LSTM ****")
        all_docs = self.batch_loader.load_all_from_numpy("all_docs")
        clf = QueryDocRegressionWrapper(CustomLSTM(300, 100),
                                        docs=all_docs, proportion_neg_examples=-1, 
                                        class_weights=self.batch_loader.get_class_weights(),
                                        matrix_embeddings=True)

        n_epoch = 1
        max_epoch = 10
        while True:
            partial_X_train, partial_y_train = self.batch_loader.get_next_batch()
            if partial_X_train is None or partial_y_train is None:
                if n_epoch >= max_epoch:
                    break

                n_epoch += 1
                self.batch_loader.next_epoch()

            clf.partial_fit(partial_X_train, partial_y_train)

        X_train, X_valid, y_train, y_valid = self.batch_loader.get_X_and_y()

        print("Coveo score on train : {}".format(clf.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(clf.score(X_valid, y_valid)))

