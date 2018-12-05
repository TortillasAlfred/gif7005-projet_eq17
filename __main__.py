from experiments.query_doc import PoC
from experiments.LR import LR

if __name__ == "__main__":
    exp = PoC(load_from_numpy=True)
    # exp = LR(load_from_numpy=False)
    exp.run_experiment()
