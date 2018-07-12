import numpy as np
import re

def accuracy(predictions, target):
    return np.mean(np.equal(predictions, target).astype(np.float32))

def decontract(str):
    # ret = str.replace("'re ", " are ")
    # ret = ret.replace("'RE ", " ARE ")
    ret = re.sub(r'\'re[ .?!,:;]', lambda x: " are" + x.group(0)[3:], str)
    ret = re.sub(r'\'RE[ .?!,:;]', lambda x: " ARE" + x.group(0)[3:], ret)

    # ret = ret.replace("'m ", " am ")
    # ret = ret.replace("'M ", " AM ")
    ret = re.sub(r'\'m[ .?!,:;]', lambda x: " am" + x.group(0)[2:], ret)
    ret = re.sub(r'\'M[ .?!,:;]', lambda x: " AM" + x.group(0)[2:], ret)


    ret = ret.replace("won't", "will n't")
    ret = ret.replace("WON'T", "WILL N'T")
    ret = ret.replace("ain't", "not")
    ret = ret.replace("AIN'T", "NOT")
    ret = re.sub(r'n\'t[ .?!,:;]', lambda x: " not" + x.group(0)[3:], ret)
    ret = re.sub(r'N\'T[ .?!,:;]', lambda x: " NOT" + x.group(0)[3:], ret)



    # ret = ret.replace("'ve ", " have ")
    # ret = ret.replace("'VE ", " HAVE ")
    ret = re.sub(r'\'ve[ .?!,:;]', lambda x: " have" + x.group(0)[3:], ret)
    ret = re.sub(r'\'VE[ .?!,:;]', lambda x: " HAVE" + x.group(0)[3:], ret)


    # ret = ret.replace("'ll ", " will ")
    # ret = ret.replace("'LL ", " WILL ")
    ret = re.sub(r'\'ll[ .?!,:;]', lambda x: " will" + x.group(0)[3:], ret)
    ret = re.sub(r'\'LL[ .?!,:;]', lambda x: " WILL" + x.group(0)[3:], ret)


    # ret = ret.replace("'s ", " 's ")
    # ret = ret.replace("'S ", " 'S ")
    ret = re.sub(r'\'s[ .?!,:;]', lambda x: " 's " + x.group(0)[2:], ret)
    ret = re.sub(r'\'S[ .?!,:;]', lambda x: " 'S " + x.group(0)[2:], ret)
    # ret = re.sub(r'\'s[ .?!,:;]', lambda x: " <apos> " + x.group(0)[2:], ret)
    # ret = re.sub(r'\'S[ .?!,:;]', lambda x: " <upp> <apos> " + x.group(0)[2:], ret)

    ret = re.sub(r'\'d[ .?!,:;]', lambda x: " 'd " + x.group(0)[2:], ret)
    ret = re.sub(r'\'D[ .?!,:;]', lambda x: " 'D " + x.group(0)[2:], ret)


    return ret


def preprocess(X_raw, word2idx, UNKNOWN_TOKEN, seq_len, stopwords_list = None):
    X = np.zeros((len(X_raw), seq_len), dtype = np.int32)
    len_array = []
    for ind in range(X_raw.shape[0]):
        # Remove space characters:
        X_raw[ind] = re.sub(r'[\\][tnrfv]', ' ', X_raw[ind]).lower()
        X_raw[ind] = re.sub(r'[\\]xa0', ' ', X_raw[ind])
        X_raw[ind] = re.sub(r'[\\]+[^\w]', ' ', X_raw[ind])
        X_raw[ind] = re.sub(r' +', ' ', X_raw[ind])
        X_raw[ind] = decontract(X_raw[ind])
        X_raw[ind] = re.sub(r' [A-Z]{2,}', lambda x: " <upp>" + x.group(0), X_raw[ind])
        X_raw[ind] = re.sub(r'[^a-zA-Z][0-9]+', lambda x: " <num>", X_raw[ind])
        word_array = re.sub(r'[^\w \'<>]', '', X_raw[ind]).lower().split()
        if stopwords_list is not None:
            word_array = [word for word in word_array if word not in stopwords_list]
        # print(word_array)
        # if len(word_array) == 2386:
        #     print(ind)
        if len(word_array) > 0:
            word_array = [word2idx.get(word, UNKNOWN_TOKEN) for word in word_array]
            X[ind, -len(word_array):] = np.array(word_array).astype(np.int32)

    return X
