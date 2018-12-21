from loading.dataLoader import DataLoader
from loading.bagOfWordsVectorizer import BagOfWordsVectorizer
from loading.wordVectorizer import MatrixWordVectorizer
from loading.oneHotEncoder import OneHotEncoder
from learners.neural_network_cosine_classifier import Neural_network_cosine_classifier

class two_level_classifier:
    def __init__(self):
        vectBOW = BagOfWordsVectorizer()
        vectWV = MatrixWordVectorizer()
        enc = OneHotEncoder()
        data_folder_path = "./data/"
        load_from_numpy = True
        self.loader = DataLoader(vectorizer=vectBOW, one_hot_encoder=enc,
                                 search_features=DataLoader.default_search_features,
                                 click_features=DataLoader.default_click_features,
                                 data_folder_path=data_folder_path, numpy_folder_path="./data/bow_oh_filtered/",
                                 load_from_numpy=load_from_numpy, filter_no_clicks=True)
        _, query_expression_embeddings, _, _, _, document_title_embeddings, all_docs_ids = DataLoader(vectorizer=vectWV,
                                                                                                      one_hot_encoder=enc,
                                                                                                      search_features=DataLoader.only_query,
                                                                                                      click_features=DataLoader.default_click_features,
                                                                                                      data_folder_path=data_folder_path,
                                                                                                      numpy_folder_path="./data/wv/",
                                                                                                      load_from_numpy=load_from_numpy,
                                                                                                      filter_no_clicks=True).load_transform_data()
        self.clf = Neural_network_cosine_classifier(data_folder_path=data_folder_path,
                                                    query_expression_embeddings=query_expression_embeddings,
                                                    document_title_embeddings=document_title_embeddings,
                                                    all_docs_ids=all_docs_ids)

    def run_experiment(self):
        _, X_valid, _, _, y_valid, _, _ = self.loader.load_transform_data()
        for i in range(1, 21):
            print("Iteration " + str(i) + " in progress")
            n_pred_neural_network = i * 20
            score = self.clf.score(X_valid, y_valid, n_pred_neural_network=n_pred_neural_network)
            f = open(str(n_pred_neural_network) + ".txt", "w+")
            f.write(str(score))
            f.close()