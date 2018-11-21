from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.dataLoader import DataLoader

from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    vect = BagOfWordsVectorizer()
    enc = OneHotEncoder()
    loader = DataLoader(vectorizer=vect, one_hot_encoder=enc, 
                        search_features=DataLoader.default_search_features,
                        click_features=DataLoader.default_click_features,
                        data_folder_path="./data/", numpy_folder_path="./data/bow_oh/", 
                        load_from_numpy=True)

    # _, _, _, _, _, _, _ = loader.load_transform_data()

    X_train, y_train = loader.load_all_from_numpy("X_train", "y_train")

    print("Loading done bby")
    clf = LinearRegression(n_jobs=-1, copy_X=False)
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))