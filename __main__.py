from experiments.query_doc import *
from experiments.LR import LR
from experiments.n_predicted_sample import GlobalExperiment
from experiments.two_level_classifier import two_level_classifier

if __name__ == "__main__":
    # all possible query_expression and document_title pairs used to train a regressor
    # exp = PoC(load_from_numpy=False)

    # Classifier based on cosine similarity
    # exp = CosineClassifiers(load_from_numpy=False)

    # Linear regression on dataset encoded with bag of words and one hot vectors
    exp = LR(load_from_numpy=False)

    # Classifiers that predict an arbitrary number of documents
    # exp = GlobalExperiment(load_from_numpy=False)

    # Linear regression based on cosine similarity
    # exp = CosineSimilarityWithLinearRegressor(load_from_numpy=False)

    # Cascading classifiers, first one is a neural network and second one is CosineMeanMax
    # exp = two_level_classifier()

    # exp.run_experiment_LR()
    # exp.run_experiment_LR_balanced()
    # exp.run_experiment_Cosine()
    # exp.run_experiment_Cosine_LR()
    exp.run_experiment()
