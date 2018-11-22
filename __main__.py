from loading.oneHotEncoder import OneHotEncoder
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.dataLoader import DataLoader
from wrappers.regression_wrapper import RegressionWrapper
from scorers.coveo_scorer import coveo_score

from sklearn.linear_model import LogisticRegressionCV, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import numpy as np

if __name__ == "__main__":
    vect = BagOfWordsVectorizer()
    enc = OneHotEncoder()
    loader = DataLoader(vectorizer=vect, one_hot_encoder=enc, 
                        search_features=DataLoader.default_search_features,
                        click_features=DataLoader.default_click_features,
                        data_folder_path="./data/", numpy_folder_path="./data/bow_oh_filtered/", 
                        load_from_numpy=True, filter_no_clicks=True)

    # _, _, _, _, _, _, _ = loader.load_transform_data()

    X_train, y_train, X_valid, y_valid, all_docs_ids = loader.load_all_from_numpy("X_train", "y_train", "X_valid", "y_valid", "all_docs_ids")

    print("Loading done bby")

    reg = RegressionWrapper(LinearRegression(), total_outputs=all_docs_ids.shape[0])
    reg.fit(X_train, y_train)
    print(reg.score(X_train, y_train))
    print(reg.score(X_valid, y_valid))
    
    # mor = MultiOutputRegressor(LogisticRegressionCV(Cs=10, cv=4, n_jobs=-1, solver='saga', tol=1e-3, max_iter=10, verbose=1))
    # mor.fit(X_train, y_train)
    # print(coveo_score(y_train, mor.predict(X_train)))
    # print(coveo_score(y_valid, mor.predict(X_valid)))
