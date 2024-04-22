import numpy as np

def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features