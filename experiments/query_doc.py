from loading.wordVectorizer import WordVectorizer
from loading.dataLoader import DataLoader
from loading.oneHotEncoder import OneHotEncoder

from wrappers.qd_regression_wrapper import QueryDocRegressionWrapper
from wrappers.regression_wrapper import RegressionWrapper

from sklearn.linear_model import LinearRegression, LogisticRegression

class PoC:
    def __init__(self, load_from_numpy):
        vectWV = WordVectorizer()
        enc = OneHotEncoder()
        self.loader_wv = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="./data/", numpy_folder_path="./data/qd_wv/",
                                    load_from_numpy=load_from_numpy, filter_no_clicks=False)

    def run_experiment(self):
        # self.run_normal_wrapped()
        self.run_qd_wrapped()

    def run_qd_wrapped_all_dataset(self):
        print("**** QD-WRAPPED PAR REG ALL DATASET ****")
        self.loader_wv.load_transform_data()
        X_train, X_valid, y_train, y_valid, all_docs = self.loader_wv.load_all_from_numpy("X_train", "X_valid",
                                                                                          "y_train", "y_valid",
                                                                                          "all_docs")

        reg = QueryDocRegressionWrapper(LogisticRegression(), all_docs, proportion_neg_examples=1)
        reg.fit(X_train, y_train)
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
