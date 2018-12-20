import pickle
from loading.dataLoader import DataLoader
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.oneHotEncoder import OneHotEncoder

class two_level_classifier:
    def __init__(self):
        self.searchEngine = self.load_model("search_engine")

    def load_model(self, model_name):
        filename = model_name + ".pck"
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model

    def run_experiment(self):
        vectBOW = BagOfWordsVectorizer()
        enc = OneHotEncoder()
        loader = DataLoader(vectorizer=vectBOW, one_hot_encoder=enc,
                            search_features=DataLoader.default_search_features,
                            click_features=DataLoader.default_click_features,
                            data_folder_path="./data/", numpy_folder_path="./data/bow_oh_filtered/",
                            load_from_numpy=True, filter_no_clicks=True)

        loader.load_transform_data()
        X_valid, y_valid, all_docs_ids = loader.load_all_from_numpy("X_valid", "y_valid", "all_docs_ids")
        X_valid = X_valid.astype(float)

        print(self.searchEngine.score(X_valid, y_valid, n_outputs=200))
