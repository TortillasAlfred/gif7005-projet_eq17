import pickle
import numpy as np
from scorers.coveo_scorer import coveo_score
from learners.cosine_similarity import MeanMaxCosineSimilarityRegressor


class Neural_network_cosine_classifier():
    def __init__(self, data_folder_path, query_expression_embeddings, document_title_embeddings, all_docs_ids):
        self.data_folder_path = data_folder_path
        self.searchEngine = self.load_model("search_engine")
        self.query_expression_embeddings = query_expression_embeddings
        self.document_title_embeddings = document_title_embeddings
        self.all_docs_ids = all_docs_ids
        self.cosine = MeanMaxCosineSimilarityRegressor(document_title_embeddings, 20, 20)

    def load_model(self, model_name):
        filename = model_name + ".pck"
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model

    def predict(self, X, n_pred_neural_network):
        neural_network_pred = self.searchEngine.predict(X, n_outputs=n_pred_neural_network)

        y_pred = np.empty((X.shape[0], 5), dtype=str)
        for i in range(X.shape[0]):
            query_expression_embedding = self.query_expression_embeddings[i]
            prediction_title_embeddings = self.document_title_embeddings[neural_network_pred[i]]

            X_regressor = np.empty(shape=(n_pred_neural_network, 40, 300), dtype="float16")
            for j in range(n_pred_neural_network):
                X_regressor[j] = np.vstack((query_expression_embedding, prediction_title_embeddings[j]))
            indices_best = self.cosine.predict(X_regressor, n_predicted_per_sample=5)
            y_pred[i] = self.all_docs_ids[neural_network_pred[i][indices_best]]

        return y_pred

    def score(self, X, y_true, n_pred_neural_network):
        y_pred = self.predict(X, n_pred_neural_network=n_pred_neural_network)
        return coveo_score(y_true, y_pred)
