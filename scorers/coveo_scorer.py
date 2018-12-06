import numpy as np


def coveo_score(y_true, y_pred):
    intersection = np.asarray([np.intersect1d(y_true[i], y_pred[i]) for i in range(len(y_true))])
    correct = np.asarray([len(col) > 0 for col in intersection])
    impossible_to_predict = np.asarray([len(col) == 0 for col in y_true])
    correct = np.ma.masked_array(correct, mask=impossible_to_predict)

    return np.mean(correct)
