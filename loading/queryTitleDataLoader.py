from loading.dataLoader import DataLoader
import pandas as pds
import numpy as np

class QueryTitleDataLoader(DataLoader):
    def __init__(self, vectorizer, data_folder_path, numpy_folder_path, load_from_numpy, load_dummy=True):
        super(QueryTitleDataLoader, self).__init__(vectorizer, None, None, None, data_folder_path, numpy_folder_path,
                                                   load_from_numpy, filter_no_clicks=False, load_dummy=load_dummy)

    def load_data_from_numpy(self):
        return self.load_all_from_numpy("X_train", "X_valid", "X_test", "y_train", "y_valid")

    def load_transform_data_from_csv(self):
        self.load_searches()
        self.load_clicks()

        self.load_transform_data()
        self.generate_labels()

        return self.load_data_from_numpy()

    def load_transform_data(self):
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

        self.nb_docs = len(tr_doc_titles)

        X_train = self.create_all_possible_combinations(tr_q_exps_train, tr_doc_titles)
        X_valid = self.create_all_possible_combinations(tr_q_exps_valid, tr_doc_titles)
        X_test = self.create_all_possible_combinations(tr_q_exps_test, tr_doc_titles)

        self.save_all_to_numpy(**{"X_train": X_train, "X_valid": X_valid, "X_test": X_test})

    def get_doc_titles(self, clicks_train, clicks_valid):
        all_clicks = pds.concat([clicks_train, clicks_valid])
        doc_titles = all_clicks["document_title"]
        doc_titles.fillna("", inplace=True)
        return np.unique(doc_titles.ravel())

    def create_all_possible_combinations(self, a, b):
        out = np.zeros((len(a) * len(b), 2), dtype=object)
        for i in range(len(a)):
            for j in range(len(b)):
                out[(i * len(b)) + j, 0] = a[i]
                out[(i * len(b)) + j, 1] = b[j]
        return out

    def generate_labels(self):
        searches_train, searches_valid, clicks_train, clicks_valid = self.load_all_from_pickle("searches_train",
                                                                                               "searches_valid",
                                                                                               "clicks_train",
                                                                                               "clicks_valid")
        doc_titles = self.get_doc_titles(clicks_train, clicks_valid)
        docs_idx = dict()
        for i, item in enumerate(doc_titles):
            docs_idx[item] = i

        y_train = np.empty(searches_train.shape[0] * self.nb_docs, dtype=bool)
        y_train.fill(False)
        y_valid = np.empty(searches_valid.shape[0] * self.nb_docs, dtype=bool)
        y_valid.fill(False)

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

        for i, clicked_docs_for_one_search in enumerate(clicked_docs_for_each_search_train):
            for doc in clicked_docs_for_one_search:
                y_train[(i * self.nb_docs) + docs_idx[doc]] = True

        for i, clicked_docs_for_one_search in enumerate(clicked_docs_for_each_search_val):
            for doc in clicked_docs_for_one_search:
                y_valid[(i * self.nb_docs) + docs_idx[doc]] = True

        self.save_all_to_numpy(**{"y_train": y_train, "y_valid": y_valid})
