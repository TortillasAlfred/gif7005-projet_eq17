from loading.dataLoader import DataLoader
import pandas as pds
import numpy as np

class QueryTitleDataLoader(DataLoader):
    def __init__(self, vectorizer, data_folder_path, numpy_folder_path, load_from_numpy, load_dummy=True):
        super(QueryTitleDataLoader, self).__init__(vectorizer, None, ["search_id", "query_expression"], ["search_id", "document_title"], data_folder_path, numpy_folder_path,
                                                   load_from_numpy, filter_no_clicks=False, load_dummy=load_dummy)

    def load_data_from_numpy(self):
        return self.load_all_from_numpy("X_train", "X_valid", "X_test", "y_train", "y_valid")

    def load_transform_data_from_csv(self):
        self.load_searches()
        self.load_clicks()

        self.transform()
        self.get_y()

        return self.load_data_from_numpy()

    def transform(self):
        searches_train, searches_valid, searches_test, clicks_train, clicks_valid = self.load_all_from_pickle("searches_train",
                                                                                                              "searches_valid",
                                                                                                              "searches_test",
                                                                                                              "clicks_train",
                                                                                                              "clicks_valid")
        tr_q_exps_train, tr_q_exps_valid, tr_q_exps_test = self.features_transformers["query_expression"](searches_train["query_expression"],
                                                                                                          searches_valid["query_expression"],
                                                                                                          searches_test["query_expression"])
        doc_titles = self.get_doc_titles(clicks_train, clicks_valid)
        tr_doc_titles = self.features_transformers["document_title"](doc_titles)[0]

        self.create_all_possible_combinations(tr_q_exps_train, tr_doc_titles, "X_train.npy")
        self.create_all_possible_combinations(tr_q_exps_valid, tr_doc_titles, "X_valid.npy")
        self.create_all_possible_combinations(tr_q_exps_test, tr_doc_titles, "X_test.npy")

    def get_doc_titles(self, clicks_train, clicks_valid):
        all_clicks = pds.concat([clicks_train, clicks_valid])
        doc_titles = all_clicks["document_title"]
        doc_titles.fillna("", inplace=True)
        return np.unique(doc_titles.ravel())

    def create_all_possible_combinations(self, a, b, filename):
        out = np.memmap(self.numpy_folder_path + filename, dtype="float32", mode="w+",
                        shape=(a.shape[0] * b.shape[0], a.shape[1] + b.shape[1]))
        for i in range(len(a)):
            for j in range(len(b)):
                out[(i * len(b)) + j] = np.hstack((a[i], b[j]))
        out.flush()

    def get_y(self):
        searches_train, searches_valid, clicks_train, clicks_valid = self.load_all_from_pickle("searches_train",
                                                                                               "searches_valid",
                                                                                               "clicks_train",
                                                                                               "clicks_valid")
        doc_titles = self.get_doc_titles(clicks_train, clicks_valid)
        docs_idx = dict()
        for i, item in enumerate(doc_titles):
            docs_idx[item] = i

        clicked_docs_for_each_search_train = []
        for s_id in searches_train.loc[:, "search_id"]:
            titles = clicks_train.loc[clicks_train.loc[:, "search_id"] == s_id, "document_title"]
            titles.fillna("", inplace=True)
            clicked_docs_for_each_search_train.append(titles.ravel())

        clicked_docs_for_each_search_val = []
        for s_id in searches_valid.loc[:, "search_id"]:
            titles = clicks_valid.loc[clicks_valid.loc[:, "search_id"] == s_id, "document_title"]
            titles.fillna("", inplace=True)
            clicked_docs_for_each_search_val.append(titles.ravel())

        y_train = np.memmap(self.numpy_folder_path + "y_train.npy", dtype=bool, mode="w+", shape=(searches_train.shape[0] * doc_titles.shape[0],))
        y_train.fill(False)
        for i, clicked_docs_for_one_search in enumerate(clicked_docs_for_each_search_train):
            for doc in clicked_docs_for_one_search:
                y_train[(i * doc_titles.shape[0]) + docs_idx[doc]] = True
        y_train.flush()

        y_valid = np.memmap(self.numpy_folder_path + "y_valid.npy", dtype=bool, mode="w+", shape=(searches_valid.shape[0] * doc_titles.shape[0],))
        y_valid.fill(False)
        for i, clicked_docs_for_one_search in enumerate(clicked_docs_for_each_search_val):
            for doc in clicked_docs_for_one_search:
                y_valid[(i * doc_titles.shape[0]) + docs_idx[doc]] = True
        y_valid.flush()
