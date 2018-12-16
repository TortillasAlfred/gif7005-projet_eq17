from experiments.query_doc import *
from experiments.LR import LR
from experiments.n_predicted_sample import GlobalExperiment

if __name__ == "__main__":
    # exp = PoC(load_from_numpy=False)
    # exp = CosineClassifiers(load_from_numpy=True)
    # exp = LR(load_from_numpy=True)
    # exp = GlobalExperiment(load_from_numpy=False)
    exp = CosineSimilarityWithLinearRegressor(load_from_numpy=True)
    # exp.run_experiment_LR()
    # exp.run_experiment_LR_balanced()
    # exp.run_experiment_Cosine()
    exp.run_experiment()    
