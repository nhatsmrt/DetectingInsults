import numpy as np
import re

def accuracy(predictions, target):
    return np.mean(np.equal(predictions, target).astype(np.float32))

def preprocess(X_raw, word2idx, UNKNOWN_TOKEN, seq_len, stopwords_list = None):
    X = np.zeros((len(X_raw), seq_len), dtype = np.int32)
    len_array = []
    for ind in range(X_raw.shape[0]):
        X_raw[ind] = re.sub(r'[\\][tnrfv]', ' ', X_raw[ind]).lower()
        X_raw[ind] = re.sub(r'[\\]xa0', ' ', X_raw[ind])
        X_raw[ind] = re.sub(r'[\\]+[^\w]', ' ', X_raw[ind])
        word_array = re.sub(r'[^\w \'\\]', '', X_raw[ind]).split()
        if stopwords_list is not None:
            word_array = [word for word in word_array if word not in stopwords_list]
        # print(word_array)
        if len(word_array) > 0:
            word_array = [word2idx.get(word, UNKNOWN_TOKEN) for word in word_array]
            X[ind, -len(word_array):] = np.array(word_array).astype(np.int32)

    return X
