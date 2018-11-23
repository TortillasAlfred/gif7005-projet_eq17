import spacy
import numpy as np


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
        transformed_data = list()
        transformed_data.append(np.array(list(self.generate_avg_word_vectors(data_train))))
        for d in data:
            transformed_data.append(np.array(list(self.generate_avg_word_vectors(d))))
        return transformed_data
