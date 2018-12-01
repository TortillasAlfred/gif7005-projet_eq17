import pandas as pds
import numpy as np
import os

from copy import deepcopy

class DataLoader:
    default_search_features = ["search_id", "search_cause", "query_expression", "query_pipeline", "facet_title", "user_type"]
    default_click_features = ["document_id", "document_title", "search_id"]
    only_query = ["search_id", "query_expression"]

    def __init__(self, vectorizer, one_hot_encoder,
                 search_features, click_features, data_folder_path,
                 numpy_folder_path, load_from_numpy, filter_no_clicks=False,
                 load_dummy=False):
        self.vectorizer = vectorizer
        self.one_hot_encoder = one_hot_encoder
        self.search_features = search_features
        self.click_features = click_features
        self.data_folder_path = data_folder_path
        self.numpy_folder_path = numpy_folder_path
        self.load_from_numpy = load_from_numpy
        self.filter_no_clicks = filter_no_clicks
        self.load_dummy = load_dummy
        if not os.path.exists(self.numpy_folder_path):
            os.makedirs(self.numpy_folder_path)
        self.features_transformers = {"search_cause": self.one_hot_transformer,
                                      "query_expression": self.vectorizer_transformer, 
                                      "query_pipeline": self.one_hot_transformer, 
                                      "facet_title": self.one_hot_transformer, 
                                      "user_type": self.one_hot_transformer, 
                                      "document_source": self.one_hot_transformer,
                                      "document_title": self.vectorizer_transformer}


    def load_transform_data(self):
        if self.load_from_numpy:
            return self.load_data_from_numpy()
        else:
            return self.load_transform_data_from_csv()


    def load_data_from_numpy(self):
        return self.load_all_from_numpy("X_train", "X_valid", "X_test",
                                        "y_train", "y_valid", "all_docs", "all_docs_ids")

    
    def save_all_to_numpy(self, **data_dict):
        for name, data in data_dict.items():
            np.save(self.numpy_folder_path + name, data)


    def load_all_from_numpy(self, *files):
        all_loaded_files = []

        for file in files:
            all_loaded_files.append(np.load(self.numpy_folder_path + file + ".npy"))

        return all_loaded_files if len(all_loaded_files) > 1 else all_loaded_files[0]

    def save_all_to_pickle(self, **data_dict):
        for name, data in data_dict.items():
            data.to_pickle(self.numpy_folder_path + name + ".pck")


    def load_all_from_pickle(self, *files):
        all_loaded_files = []

        for file in files:
            all_loaded_files.append(pds.read_pickle(self.numpy_folder_path + file + ".pck"))

        return all_loaded_files if len(all_loaded_files) > 1 else all_loaded_files[0]


    def load_transform_data_from_csv(self):
        self.load_searches()
        self.transform_searches()

        self.load_clicks()
        self.transform_clicks()

        self.get_y()

        if self.filter_no_clicks:
            self.filter_data()

        return self.load_data_from_numpy()

    
    def filter_data(self):
        X_train, X_valid, y_train, y_valid = self.load_all_from_numpy("X_train", "X_valid", "y_train", "y_valid")

        filter_train = np.where([len(col) > 0 for col in y_train])
        X_train = X_train[filter_train]
        y_train = y_train[filter_train]

        filter_valid = np.where([len(col) > 0 for col in y_valid])
        X_valid = X_valid[filter_valid]
        y_valid = y_valid[filter_valid]

        self.save_all_to_numpy(**{"X_train": X_train, "X_valid": X_valid,
                                  "y_train": y_train, "y_valid": y_valid})


    def load_searches(self):
        if self.load_dummy:
            searches_train = pds.read_csv(self.data_folder_path + "dummy_searches_train.csv")
            searches_valid = pds.read_csv(self.data_folder_path + "dummy_searches_valid.csv")
            searches_test = pds.read_csv(self.data_folder_path + "dummy_searches_test.csv")
        else:
            searches_train = pds.read_csv(self.data_folder_path + "coveo_searches_train.csv")
            searches_valid = pds.read_csv(self.data_folder_path + "coveo_searches_valid.csv")
            searches_test = pds.read_csv(self.data_folder_path + "coveo_searches_test.csv")

        self.save_all_to_pickle(**{"searches_train": searches_train[self.search_features],
                                   "searches_valid": searches_valid[self.search_features],
                                   "searches_test": searches_test[self.search_features]})


    def transform_searches(self):
        searches_train, searches_valid, searches_test = self.load_all_from_pickle("searches_train",
                                                                                 "searches_valid",
                                                                                 "searches_test")

        X_train = []
        X_valid = []
        X_test = []

        for feature in list(searches_train):
            if feature in self.features_transformers.keys():
                d_train, d_valid, d_test = self.features_transformers[feature](searches_train[feature], searches_valid[feature], searches_test[feature])
                X_train.append(d_train)
                X_valid.append(d_valid)
                X_test.append(d_test)

        X_train = np.hstack([liste for liste in X_train])
        X_valid = np.hstack([liste for liste in X_valid])
        X_test = np.hstack([liste for liste in X_test])

        self.save_all_to_numpy(**{"X_train": X_train,
                                "X_valid": X_valid,
                                "X_test": X_test})


    def load_clicks(self):
        if self.load_dummy:
            clicks_train = pds.read_csv(self.data_folder_path + "dummy_clicks_train.csv")
            clicks_valid = pds.read_csv(self.data_folder_path + "dummy_clicks_valid.csv")
        else:
            clicks_train = pds.read_csv(self.data_folder_path + "coveo_clicks_train.csv")
            clicks_valid = pds.read_csv(self.data_folder_path + "coveo_clicks_valid.csv")

        self.save_all_to_pickle(**{"clicks_train": clicks_train[self.click_features],
                                "clicks_valid": clicks_valid[self.click_features]})


    def transform_clicks(self):
        clicks_train, clicks_valid = self.load_all_from_pickle("clicks_train",
                                                               "clicks_valid")

        all_clicks = pds.concat([clicks_train, clicks_valid])
        all_clicks.drop_duplicates(subset="document_id", inplace=True)

        all_docs = []

        for feature in list(clicks_train):
            if feature in self.features_transformers.keys():
                d = self.features_transformers[feature](all_clicks[feature])
                all_docs.append(d[0])

        self.save_all_to_numpy(**{"all_docs_ids": all_clicks["document_id"].values,
                                  "all_docs": all_docs[0]})


    def get_y(self):
        searches_train, searches_valid, clicks_train, clicks_valid = self.load_all_from_pickle("searches_train",
                                                                                               "searches_valid",
                                                                                               "clicks_train",
                                                                                               "clicks_valid")
        all_docs_ids = self.load_all_from_numpy("all_docs_ids")

        all_clicks = pds.concat([clicks_train, clicks_valid])
        y_train = np.zeros((searches_train.shape[0], all_docs_ids.shape[0]), dtype=bool)
        y_valid = np.zeros((searches_valid.shape[0], all_docs_ids.shape[0]), dtype=bool)

        correspondance_train = clicks_train[["search_id", "document_id"]]
        for c in correspondance_train.values:
            y_train[np.where(searches_train.search_id.values == c[0]),\
                    np.where(all_docs_ids == c[1])] = True

        correspondance_valid = clicks_valid[["search_id", "document_id"]]
        for c in correspondance_valid.values:
            y_valid[np.where(searches_valid.search_id.values == c[0]),\
                    np.where(all_docs_ids == c[1])] = True

        y_train = np.asarray([np.where(row == 1)[0] for row in y_train])
        y_valid = np.asarray([np.where(row == 1)[0] for row in y_valid])

        self.save_all_to_numpy(**{"y_train": y_train,
                                  "y_valid": y_valid})


    def no_transformer(self, data_train, *data):
        return_data = [np.asarray(data_train)]

        for d in data:
            return_data.append(np.asarray(d))

        return return_data


    def one_hot_transformer(self, data_train, *data):
        return deepcopy(self.one_hot_encoder).fit_transform(data_train, *data)


    def vectorizer_transformer(self, data_train, *data):
        return deepcopy(self.vectorizer).fit_transform(data_train, *data)

