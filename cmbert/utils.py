import numpy as np

def avg(intervals: np.array, features: np.array) -> np.array:
    try:
        return np.average(features, axis=0)
    except:
        return features
    

def bword_vector_2_sentence(bwords, filter = [b'sp'], encoding='utf-8'):
    return ' '.join([bw.decode(encoding) for bw in bwords if bw not in filter])