import spacy
import numpy as np

from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.oneHotEncoder import OneHotEncoder
from loading.dataLoader import DataLoader

MAX_QUERY_LENGTH = 20


class WordVectorizer(object):
    def __init__(self):
        self.nlp = spacy.load("en_vectors_web_lg")

    def generate_matrix_word_vectors(self, queries):
        '''
        :param queries: iterator of queries in NL
        :return: matrix with (nb_queries x MAX_QUERY_LENGTH x vector size(300) ) size
        '''
        queries = np.array(queries)
        wv = np.zeros((queries.size, MAX_QUERY_LENGTH, 300))
        for i in range(queries.size):
            doc = self.nlp(str(queries[i]))
            j = 0
            for token in doc:
                if token.has_vector and j < MAX_QUERY_LENGTH:
                    wv[i, j] = token.vector
                j += 1
        return wv

    def generate_dict_word_vectors(self, queries):
        '''
        :param queries: iterator of queries in NL
        :return: dictionary with query as key and list of word embeddings as value
        '''
        wv = {}
        for query in queries:
            doc = self.nlp(str(query))
            exp = []
            for token in doc:
                if token.has_vector:
                    exp.append(token.vector)
            wv[query] = exp
        return wv

    def generate_dict_sentence_vectors_spacy(self, queries):
        '''
        :param queries: iterator of queries in NL
        :return: dictionary with query as key and sentence embedding as value (made by spacy)
        '''

        wv = {}
        for query in queries:
            doc = self.nlp(str(query))
            wv[query] = doc.vector
        return wv

    def generate_dict_sentence_vectors_avghandmade(self, queries):
        '''
        :param queries: iterator of queries in NL
        :return: dictionary with query as key and sentence embedding as value (with handmade mean of spacy word vectors)
        '''

        wv = {}
        for query in queries:
            doc = self.nlp(str(query))
            exp = list()
            for token in doc:
                if token.has_vector:
                    exp.append(token.vector)
            if not exp:
                wv[query] = np.zeros(300)
            else:
                wv[query] = np.mean(exp, axis=0)
        return wv


class MatrixWordVectorizer(WordVectorizer):
    def __init__(self):
        super(MatrixWordVectorizer, self).__init__()

    def fit_transform(self, data_train, *data):
        transformed_data = list()
        transformed_data.append(np.array(list(self.generate_matrix_word_vectors(data_train))))
        for d in data:
            transformed_data.append(np.array(list(self.generate_matrix_word_vectors(d))))
        return transformed_data


class DictWordVectorizer(WordVectorizer):
    def __init__(self):
        super(DictWordVectorizer, self).__init__()

    def fit_transform(self, data_train, *data):
        transformed_data = list()
        transformed_data.append(np.array(list(self.generate_dict_word_vectors(data_train))))
        for d in data:
            transformed_data.append(np.array(list(self.generate_dict_word_vectors(d))))
        return transformed_data


class DictSentenceVectorizerSpacy(WordVectorizer):
    def __init__(self):
        super(DictSentenceVectorizerSpacy, self).__init__()

    def fit_transform(self, data_train, *data):
        transformed_data = list()
        transformed_data.append(np.array(list(self.generate_dict_sentence_vectors_spacy(data_train))))
        for d in data:
            transformed_data.append(np.array(list(self.generate_dict_sentence_vectors_spacy(d))))
        return transformed_data


class DictSentenceVectorizerHM(WordVectorizer):
    def __init__(self):
        super(DictSentenceVectorizerHM, self).__init__()

    def fit_transform(self, data_train, *data):
        transformed_data = list()
        transformed_data.append(np.array(list(self.generate_dict_sentence_vectors_avghandmade(data_train))))
        for d in data:
            transformed_data.append(np.array(list(self.generate_dict_sentence_vectors_avghandmade(d))))
        return transformed_data


if __name__== "__main__":
    vect = BagOfWordsVectorizer()
    enc = OneHotEncoder()
    loader = DataLoader(vectorizer=vect, one_hot_encoder=enc,
                        search_features=DataLoader.default_search_features,
                        click_features=DataLoader.default_click_features,
                        data_folder_path="../data/", numpy_folder_path="../data/bow_oh/",
                        load_from_numpy=False)

    loader.load_transform_data()
    searches_train = loader.load_all_from_pickle("searches_train")

    test1 = MatrixWordVectorizer()
    test1.fit_transform(searches_train.query_expression.values)

    test2 = DictWordVectorizer()
    test2.fit_transform(searches_train.query_expression.values)

    test3 = DictSentenceVectorizerSpacy()
    test3.fit_transform(searches_train.query_expression.values)

    test4 = DictSentenceVectorizerHM()
    test4.fit_transform(searches_train.query_expression.values)
