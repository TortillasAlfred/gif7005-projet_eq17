from sklearn.neighbors import KNeighborsClassifier

from loading.wordVectorizer import *
from loading.dataLoader import DataLoader
from loading.oneHotEncoder import OneHotEncoder


vectWV = DictSentenceVectorizerSpacy()
enc = OneHotEncoder()
loader = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="../data/", numpy_folder_path="../data/qd_wv/",
                                    load_from_numpy=True, filter_no_clicks=False)

loader.load_transform_data()
training_data = loader.load_all_from_numpy("test_train")
validation_data = loader.load_all_from_numpy("test_valid")

X_train = training_data[:,0]
y_train = (np.sum(training_data[:, 1], axis=1)*1000)
y_train = list(map(int, y_train))

X_valid = validation_data[:,0]
y_valid = np.sum(validation_data[:,1], axis=1)*1000
y_valid = list(map(int, y_valid))

clf = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='auto')
clf.fit(X_train, y_train)

for query, doc in validation_data:
    pass

print(clf.score(validation_data[:, 0], validation_data[:, 1]))