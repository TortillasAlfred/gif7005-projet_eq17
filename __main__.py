from experiments.query_doc import PoC

if __name__ == "__main__":
    exp = PoC(load_from_numpy=False)
    exp.run_experiment()
