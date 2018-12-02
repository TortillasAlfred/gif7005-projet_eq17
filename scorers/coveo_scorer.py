import numpy as np


def coveo_score(y_true, y_pred):
    y_true, y_pred = convert_to_classes(y_true, y_pred)

    intersection = np.asarray([np.intersect1d(y_true[i], y_pred[i]) for i in range(len(y_true))])
    correct = np.asarray([len(col) > 0 for col in intersection])
    impossible_to_predict = np.asarray([len(col) == 0 for col in y_true])
    correct = np.ma.masked_array(correct, mask=impossible_to_predict)

    return np.mean(correct)
    
def convert_to_classes(*y):
    all_y = []

    for y_set in y:
        if y_set.ndim == 1 or y_set.shape[1] <= 5:
            all_y.append(y_set)
        else:
            all_y.append([np.where(row == 1) for row in y_set])

    return all_y
