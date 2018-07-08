import numpy as np

def accuracy(predictions, target):
    return np.mean(np.equal(predictions, target).astype(np.float32))