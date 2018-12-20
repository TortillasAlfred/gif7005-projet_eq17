from sklearn.neighbors import KNeighborsClassifier

from loading.wordVectorizer import *
from loading.dataLoader import DataLoader
from loading.oneHotEncoder import OneHotEncoder

from loading.bagOfWordsVectorizer import BagOfWordsVectorizer


vectWV = BagOfWordsVectorizer()
enc = OneHotEncoder()
loader = DataLoader(vectorizer=vectWV, one_hot_encoder=enc,
                                    search_features=DataLoader.default_search_features,
                                    click_features=DataLoader.default_click_features,
                                    data_folder_path="../data/", numpy_folder_path="../data/bow_oh_filtered/",
                                    load_from_numpy=True, filter_no_clicks=False)

loader.load_transform_data()
X_train = loader.load_all_from_numpy("X_train")
X_valid = loader.load_all_from_numpy("X_valid")
y_train = loader.load_all_from_numpy("y_train")
y_valid = loader.load_all_from_numpy("y_valid")

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

'''X_train = training_data[:,0]
y_train = (np.sum(training_data[:, 1], axis=1)*1000)
y_train = list(map(int, y_train))

X_valid = validation_data[:,0]
y_valid = np.sum(validation_data[:,1], axis=1)*1000
y_valid = list(map(int, y_valid))'''

for n in [1,3,10,30]:
    clf = KNeighborsClassifier(n_neighbors=n, weights='distance', algorithm='auto')
    clf.fit(X_train, y_train)

    predicts = clf.predict(X_valid)
    kneighbors = clf.kneighbors(X_valid, 4, return_distance=False)

    predicted_neighbors = np.zeros((kneighbors.shape[0], kneighbors.shape[1]+1))

    good = 0
    for i in range(len(kneighbors)):
        predicted_neighbors[i][0] = predicts[i]
        for j in range(len(kneighbors[i])):
            predicted_neighbors[i][j+1] = y_train[kneighbors[i,j]]
        if y_valid[i] in predicted_neighbors[i]:
            good += 1

    print(good)
