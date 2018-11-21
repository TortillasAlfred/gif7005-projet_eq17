from sklearn.feature_extraction.text import CountVectorizer

from nltk import word_tokenize

class BagOfWordsVectorizer:
    def __init__(self, min_freq=3,
                 tokenizer=word_tokenize,
                 excluded_values=['``', '\'\'', ':', '@', "'", '!', '#', '(', ')', '-', '.', '/',
                                  ';', '<', '=', '>', ',', '?', '[', ']']):
        self.count_vectorizer = CountVectorizer(min_df=min_freq, tokenizer=tokenizer,
                                                stop_words=excluded_values)

    
    def fit_transform(self, data_train, *data):
        self.fill(data_train)
        self.fit(data_train)

        transformed_data = []
        transformed_data.append(self.transform(data_train))

        for d in data:
            self.fill(d)
            transformed_data.append(self.transform(d))

        return transformed_data


    def fill(self, data):
        data.fillna("", inplace=True)
        

    def fit(self, data_train):
        self.count_vectorizer.fit(data_train.values)


    def transform(self, data):
        data_bow = self.count_vectorizer.transform(data.values).toarray()
        data_bow[data_bow > 0] = 1

        return data_bow
    