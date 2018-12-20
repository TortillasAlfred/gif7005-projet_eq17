import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from loading.dataLoader import DataLoader
from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer

from wrappers.knn_wrapper import KNNWrapper

class KNN:
    def __init__(self):

        vectWV = BagOfWordsVectorizer()
        enc = OneHotEncoder()
        self.loader = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                            search_features=DataLoader.default_search_features,
                                            click_features=DataLoader.default_click_features,
                                            data_folder_path="../data/", numpy_folder_path="../data/bow_oh_filtered/",
                                            load_from_numpy=True, filter_no_clicks=False)

    def preprocess_data(self):
        X_train, X_valid, _, y_train, y_valid, _, _ = self.loader.load_transform_data()

        for i in range(len(y_train)):
            while len(y_train[i]) > 1:
                X_train = np.append(X_train, [X_train[i]], axis=0)
                y_train = np.append(y_train, y_train[i][0])
                y_train[i] = np.delete(y_train[i], 0)
        y_train = np.array(list(map(int, y_train)))

        for i in range(len(y_valid)):
            while len(y_valid[i]) > 1:
                X_valid = np.append(X_valid, [X_valid[i]], axis=0)
                y_valid = np.append(y_valid, y_valid[i][0])
                y_valid[i] = np.delete(y_valid[i], 0)
        y_valid = np.array(list(map(int, y_valid)))

        return X_train, X_valid, y_train, y_valid

    def run_filtered(self):
        print("**** FILTERED EXP ****")
        X_train, X_valid, y_train, y_valid = self.preprocess_data()

        knn = KNNWrapper(n_neighbors=17, weights='distance', algorithm='auto')
        knn.fit(X_train, y_train)

        print("Coveo score on train : {}".format(knn.score(X_train, y_train)))
        print("Coveo score on valid : {}".format(knn.score(X_valid, y_valid)))


if __name__ == "__main__":
    test = KNN()
    test.run_filtered()




