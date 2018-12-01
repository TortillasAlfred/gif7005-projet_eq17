from loading.wordVectorizer import WordVectorizer
from loading.queryTitleDataLoader import QueryTitleDataLoader

from sklearn.linear_model import LinearRegression

class PoC:
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
