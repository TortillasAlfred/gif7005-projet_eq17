import spacy
import numpy as np

from loading.bagOfWordsVectorizer import BagOfWordsVectorizer

from loading.oneHotEncoder import OneHotEncoder
from loading.dataLoader import DataLoader

MAX_QUERY_LENGTH = 20


class WordVectorizer(object):
    def __init__(self):
        self.nlp = spacy.load("en_vectors_web_lg")

    def generate_dict_word_vectors(self, queries):
        wv = {}
        for query in queries:
            doc = self.nlp(str(query))
            exp = []
            for token in doc:
                if token.has_vector:
                    exp.append(token.vector)
            wv[query] = exp
        return wv

    def generate_matrix_word_vectors(self, queries):
        queries = np.array(queries)
        wv = np.zeros((queries.size, MAX_QUERY_LENGTH, 300))
        for i in range(queries.size):
            doc = self.nlp(str(queries[i]))
            j=0
            for token in doc:
                if token.has_vector and j < MAX_QUERY_LENGTH:
                    wv[i, j] = token.vector
                j += 1
        return wv

    def generate_sentence_vector(self, queries):
        queries = np.array(queries)
        wv = np.zeros((queries.size, 300))
        for i in range(queries.size):
            doc = self.nlp(str(queries[i]))
            wv[i] = doc.vector
        return wv

    def generate_avg_word_vectors(self, queries):
        queries = np.array(queries)
        wv = np.zeros([queries.size, 300])
        for i in range(queries.size):
            doc = self.nlp(str(queries[i]))
            exp = list()
            for token in doc:
                if token.has_vector:
                    exp.append(token.vector)
            if not exp:
                wv[i] = np.zeros(300)
            else:
                wv[i] = np.mean(exp, axis=0)
        return wv

    def fit_transform(self, data_train, *data):
        self.nlp = spacy.load("en_vectors_web_lg")
        transformed_data = list()
        transformed_data.append(np.array(list(self.generate_avg_word_vectors(data_train))))
        for d in data:
            transformed_data.append(np.array(list(self.generate_avg_word_vectors(d))))
        return transformed_data


if __name__=="__main__":

    vect = BagOfWordsVectorizer()
    enc = OneHotEncoder()
    loader = DataLoader(vectorizer=vect, one_hot_encoder=enc,
                        search_features=DataLoader.default_search_features,
                        click_features=DataLoader.default_click_features,
                        data_folder_path="../data/", numpy_folder_path="../data/bow_oh/",
                        load_from_numpy=False)

    loader.load_transform_data()
    searches_train = loader.load_all_from_pickle("searches_train")
    WV = WordVectorizer()
    test = WV.generate_matrix_word_vectors(searches_train.query_expression.values)
    test2 = WV.generate_sentence_vector(searches_train.query_expression.values)
    print("test")
