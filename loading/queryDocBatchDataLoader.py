from loading.dataLoader import DataLoader
import pandas as pds
import numpy as np
import itertools


class QueryDocBatchDataLoader(DataLoader):
    def __init__(self, vectorizer, encoder, batch_size, data_folder_path, numpy_folder_path,
                 load_from_numpy, filter_no_clicks, load_dummy=True, generate_pairs=False):
        super(QueryDocBatchDataLoader, self).__init__(vectorizer=vectorizer, one_hot_encoder=encoder,
                                                      search_features=DataLoader.default_search_features,
                                                      click_features=DataLoader.default_click_features,
                                                      data_folder_path=data_folder_path,
                                                      numpy_folder_path=numpy_folder_path,
                                                      load_from_numpy=load_from_numpy, filter_no_clicks=filter_no_clicks,
                                                      load_dummy=load_dummy)
        self.tr_q_exps_train, self.tr_q_exps_valid, self.tr_q_exps_test = self.__load_transform_queries()
        self.tr_doc_titles = self.__load_transform_doc_titles()
        self.tr_y, self.valid_y = self.__load_transform_y()
        if filter_no_clicks:
            self.filter_data()
        if generate_pairs:
            self.pairs = self.__generate_pairs()
        else:
            self.pairs = self.__load_pairs()
        self.batch_size = batch_size
        self.current_batch = 0
        self.current_epoch = 1
        self.num_batches = int((self.tr_q_exps_train.shape[0] * self.tr_doc_titles.shape[0]) / self.batch_size)

    def __load_transform_queries(self):
        self.load_searches()
        searches_train, searches_valid, searches_test, = self.load_all_from_pickle("searches_train",
                                                                                   "searches_valid",
                                                                                   "searches_test")
        return self.features_transformers["query_expression"](searches_train["query_expression"],
                                                              searches_valid["query_expression"],
                                                              searches_test["query_expression"])

    def __load_transform_doc_titles(self):
        self.load_clicks()
        clicks_train, clicks_valid = self.load_all_from_pickle("clicks_train", "clicks_valid")
        doc_titles = self.__get_doc_titles(clicks_train, clicks_valid)
        return self.features_transformers["document_title"](doc_titles)[0]

    def __load_transform_y(self):
        self.get_y()
        return self.load_all_from_numpy("y_train"), self.load_all_from_numpy("y_valid")

    def __get_doc_titles(self, clicks_train, clicks_valid):
        all_clicks = pds.concat([clicks_train, clicks_valid])
        doc_titles = all_clicks["document_title"]
        doc_titles.fillna("", inplace=True)
        return np.unique(doc_titles.ravel())

    def __generate_pairs(self):
        queries_idx = np.asarray(range(self.tr_q_exps_train.shape[0]))
        docs_idx = np.asarray(range(self.tr_doc_titles.shape[0]))
        combinations = np.memmap(self.numpy_folder_path + "random_pairs_filtered.npy", dtype=np.uint32, mode="w+",
                                 shape=(self.tr_q_exps_train.shape[0] * self.tr_doc_titles.shape[0], 2))
        product = itertools.product(queries_idx, docs_idx)
        counter = 0
        for pair in product:
            combinations[counter] = pair
            counter += 1
        np.random.shuffle(combinations)
        return combinations

    def __load_pairs(self):
        return np.memmap(self.numpy_folder_path + "random_pairs_filtered.npy", dtype=np.uint32, mode='r',
                  shape=(self.tr_q_exps_train.shape[0] * self.tr_doc_titles.shape[0], 2))

    def get_next_batch(self):
        next_batch = (None, None)
        if self.current_batch <= self.num_batches:
            start_idx = int(self.current_batch * self.batch_size)
            end_idx = int(start_idx + self.batch_size)
            next_pairs = self.pairs[start_idx:end_idx]
            next_batch = self.__get_batch(next_pairs)
            self.current_batch += 1

        print("Epoch {}, Batch {}/{}".format(self.current_epoch, self.current_batch, self.num_batches))
        return next_batch

    def __get_batch(self, pairs):
        X = np.empty(shape=(pairs.shape[0],self.tr_q_exps_train.shape[1] + self.tr_doc_titles.shape[1]), dtype="float16")
        y = np.empty(shape=(pairs.shape[0], ), dtype=bool)

        for i, pair in enumerate(pairs):
            X[i] = np.hstack((self.tr_q_exps_train[pair[0]], self.tr_doc_titles[pair[1]]))
            y[i] = pair[1] in self.tr_y[pair[0]]

        return X, y

    def filter_data(self):
        filter_train = np.where([len(col) > 0 for col in self.tr_y])
        self.tr_q_exps_train = self.tr_q_exps_train[filter_train]
        self.tr_y = self.tr_y[filter_train]

        filter_valid = np.where([len(col) > 0 for col in self.valid_y])
        self.tr_q_exps_valid = self.tr_q_exps_valid[filter_valid]
        self.valid_y = self.valid_y[filter_valid]


    def next_epoch(self):
        self.current_batch = 0
        self.current_epoch += 1

    def get_class_weights(self):
        n_pos = np.sum([len(col) for col in self.tr_y])
        n_samples = self.tr_y.shape[0] * self.tr_doc_titles.shape[0]
        n_neg = n_samples - n_pos

        weight_pos = n_samples/(2 * n_pos)
        weight_neg = n_samples/(2 * n_neg)

        return {False: weight_neg, True: weight_pos}

    def get_X_and_y(self):
        return self.tr_q_exps_train, self.tr_q_exps_valid, self.tr_y, self.valid_y
