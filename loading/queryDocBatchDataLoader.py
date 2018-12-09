from loading.dataLoader import DataLoader
import pandas as pds
import numpy as np
import itertools


class QueryDocBatchDataLoader(DataLoader):
    def __init__(self, vectorizer, batch_size, data_folder_path, numpy_folder_path,
                 load_from_numpy, load_dummy=True, generate_pairs=False):
        super(QueryDocBatchDataLoader, self).__init__(vectorizer=vectorizer, one_hot_encoder=None,
                                                      search_features=DataLoader.default_search_features,
                                                      click_features=DataLoader.default_click_features,
                                                      data_folder_path=data_folder_path,
                                                      numpy_folder_path=numpy_folder_path,
                                                      load_from_numpy=load_from_numpy, filter_no_clicks=False,
                                                      load_dummy=load_dummy)
        self.tr_q_exps_train, self.tr_q_exps_valid, self.tr_q_exps_test = self.load_transform_queries()
        self.tr_doc_titles = self.load_transform_doc_titles()
        if generate_pairs:
            self.pairs = self.generate_pairs()
        else:
            self.pairs = self.load_pairs()
        self.batch_size = batch_size
        self.current_batch = 0

    def load_transform_queries(self):
        self.load_searches()
        searches_train, searches_valid, searches_test, = self.load_all_from_pickle("searches_train",
                                                                                   "searches_valid",
                                                                                   "searches_test")
        return self.features_transformers["query_expression"](searches_train["query_expression"],
                                                              searches_valid["query_expression"],
                                                              searches_test["query_expression"])

    def load_transform_doc_titles(self):
        self.load_clicks()
        clicks_train, clicks_valid = self.load_all_from_pickle("clicks_train", "clicks_valid")
        doc_titles = self.get_doc_titles(clicks_train, clicks_valid)
        return self.features_transformers["document_title"](doc_titles)[0]

    def get_doc_titles(self, clicks_train, clicks_valid):
        all_clicks = pds.concat([clicks_train, clicks_valid])
        doc_titles = all_clicks["document_title"]
        doc_titles.fillna("", inplace=True)
        return np.unique(doc_titles.ravel())

    def generate_pairs(self):
        queries_idx = np.asarray(range(self.tr_q_exps_train.shape[0]))
        docs_idx = np.asarray(range(self.tr_doc_titles.shape[0]))
        combinations = np.memmap(self.numpy_folder_path + "random_pairs.npy", dtype=np.uint32, mode="w+",
                                 shape=(self.tr_q_exps_train.shape[0] * self.tr_doc_titles.shape[0], 2))
        product = itertools.product(queries_idx, docs_idx)
        counter = 0
        for pair in product:
            combinations[counter] = pair
            counter += 1
        np.random.shuffle(combinations)
        return combinations

    def load_pairs(self):
        return np.memmap(self.numpy_folder_path + "random_pairs.npy", dtype=np.uint32, mode='r',
                  shape=(self.tr_q_exps_train.shape[0] * self.tr_doc_titles.shape[0], 2))

    def get_next_batch(self):
        next_batch = (None, None)
        if self.current_batch < (self.tr_q_exps_train.shape[0] * self.tr_doc_titles.shape[0]) / self.batch_size:
            start_idx = self.current_batch * self.batch_size
            end_idx = start_idx + self.batch_size
            next_pairs = self.pairs[start_idx:end_idx]
            next_batch = (self.get_batch_X(next_pairs), self.get_batch_y(next_pairs))
            self.current_batch += 1
        return next_batch

    def get_batch_X(self, pairs):
        X = np.empty(shape=(pairs.shape[0],self.tr_q_exps_train.shape[1] + self.tr_doc_titles.shape[1]), dtype="float16")
        for i, pair in enumerate(pairs):
            X[i] = np.hstack((self.tr_q_exps_train[pair[0]], self.tr_doc_titles[pair[1]]))
        return X

    def get_batch_y(self, pairs):
        searches_train, clicks_train, clicks_valid = self.load_all_from_pickle("searches_train", "clicks_train", "clicks_valid")

        doc_titles = self.get_doc_titles(clicks_train, clicks_valid)
        clicked_doc_titles_for_each_search = []
        for s_id in searches_train.loc[:, "search_id"]:
            titles = clicks_train.loc[clicks_train.loc[:, "search_id"] == s_id, "document_title"]
            titles.fillna("", inplace=True)
            clicked_doc_titles_for_each_search.append(titles.ravel())

        y = np.zeros(shape=(pairs.shape[0],), dtype=bool)
        for i, pair in enumerate(pairs):
            if doc_titles[pair[1]] in clicked_doc_titles_for_each_search[pair[0]]:
                y[i] = True
        return y
