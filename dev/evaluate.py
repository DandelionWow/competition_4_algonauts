from scipy.stats import pearsonr as corr
import numpy as np

def calculate_pearson_metric(valid, pred):
    result = np.zeros(valid.shape[0])
    if valid.shape[1] != 0 and pred.shape[1] != 0:
        for v in range(valid.shape[0]):
            result[v] = corr(valid[v], pred[v])[0]
    return round(result.mean(),4)