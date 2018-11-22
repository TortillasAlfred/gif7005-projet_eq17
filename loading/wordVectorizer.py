import spacy
import numpy as np
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer

from loading.oneHotEncoder import OneHotEncoder
from loading.dataLoader import DataLoader


class WordVectorizer(object):
    def __init__(self):
        self.nlp = spacy.load("en_vectors_web_lg")


    def generate_list_word_vectors(self, queries):
        wv = {}
        for query in queries:
            doc = self.nlp(str(query))
            exp = []
            for token in doc:
                if token.has_vector:
                    exp.append(token.vector)
            wv[query] = exp
        return wv

    def generate_avg_word_vectors(self, queries):
        wv = {}
        for query in queries:
            doc = self.nlp(str(query))
            exp = []
            for token in doc:
                if token.has_vector:
                    exp.append(token.vector)
            wv[query] = np.mean(exp, axis=0)
        return wv

if __name__=="__main__":

    vect = BagOfWordsVectorizer()
    enc = OneHotEncoder()
    loader = DataLoader(vectorizer=vect, one_hot_encoder=enc,
                        search_features=DataLoader.default_search_features,
                        click_features=DataLoader.default_click_features,
                        data_folder_path="../data/", numpy_folder_path="../data/bow_oh/",
                        load_from_numpy=True)

    searches_train = loader.load_all_from_pickle("searches_train")
    WV = WordVectorizer()
    list_wv = WV.generate_list_word_vectors(searches_train.query_expression.values)
    avg_wv = WV.generate_avg_word_vectors(searches_train.query_expression.values)
