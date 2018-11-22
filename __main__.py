from experiments.LR import LR

if __name__ == "__main__":
    exp = LR(load_from_numpy=False)
    exp.run_experiment()
