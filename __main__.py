from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.dataLoader import DataLoader

if __name__ == "__main__":
    vect = BagOfWordsVectorizer()
    enc = OneHotEncoder()
    loader = DataLoader(vectorizer=vect, one_hot_encoder=enc, 
                        search_features=DataLoader.default_search_features,
                        click_features=DataLoader.default_click_features,
                        data_folder_path="./data/", numpy_folder_path="./data/bow_oh/", 
                        load_from_numpy=False)

    X_train, X_valid, X_test, y_train, y_valid, all_docs_ids = loader.load_transform_data()
    