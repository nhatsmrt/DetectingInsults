import numpy as np
import re
from textblob import TextBlob
from yandex_translate import YandexTranslate, YandexTranslateException
import pandas as pd

def accuracy(predictions, target):
    return np.mean(np.equal(predictions, target).astype(np.float32))

def decontract(str):
    ret = re.sub(r'\'re[ .?!,:;]', lambda x: " are" + x.group(0)[3:], str)
    ret = re.sub(r'\'RE[ .?!,:;]', lambda x: " ARE" + x.group(0)[3:], ret)

    ret = re.sub(r'\'m[ .?!,:;]', lambda x: " am" + x.group(0)[2:], ret)
    ret = re.sub(r'\'M[ .?!,:;]', lambda x: " AM" + x.group(0)[2:], ret)


    ret = ret.replace("won't", "will n't")
    ret = ret.replace("WON'T", "WILL N'T")
    ret = ret.replace("ain't", "not")
    ret = ret.replace("AIN'T", "NOT")
    ret = re.sub(r'n\'t[ .?!,:;]', lambda x: " not" + x.group(0)[3:], ret)
    ret = re.sub(r'N\'T[ .?!,:;]', lambda x: " NOT" + x.group(0)[3:], ret)



    ret = re.sub(r'\'ve[ .?!,:;]', lambda x: " have" + x.group(0)[3:], ret)
    ret = re.sub(r'\'VE[ .?!,:;]', lambda x: " HAVE" + x.group(0)[3:], ret)


    ret = re.sub(r'\'ll[ .?!,:;]', lambda x: " will" + x.group(0)[3:], ret)
    ret = re.sub(r'\'LL[ .?!,:;]', lambda x: " WILL" + x.group(0)[3:], ret)


    ret = re.sub(r'\'s[ .?!,:;]', lambda x: " 's " + x.group(0)[2:], ret)
    ret = re.sub(r'\'S[ .?!,:;]', lambda x: " 'S " + x.group(0)[2:], ret)

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
        # X_raw[ind] = re.sub(r'(\.\.\.)|[.,?!:;=-]', lambda x: " " + x.group(0) + " ", X_raw[ind])
        X_raw[ind] = re.sub(r' +', ' ', X_raw[ind])
        X_raw[ind] = decontract(X_raw[ind])
        X_raw[ind] = re.sub(r' [A-Z]{2,}', lambda x: " <upp>" + x.group(0), X_raw[ind])
        X_raw[ind] = re.sub(r'[^a-zA-Z][0-9]+', lambda x: " <num>", X_raw[ind])
        word_array = re.sub(r'[^\w \'<>]', '', X_raw[ind]).lower().split()
        # word_array = re.sub(r'[^\w \'<>.,?!:;=-]', '', X_raw[ind]).lower().split()
        if stopwords_list is not None:
            word_array = [word for word in word_array if word not in stopwords_list]
        # print(word_array)
        # if len(word_array) == 2386:
        #     print(ind)
        if len(word_array) > 0:
            word_array = [word2idx.get(word, UNKNOWN_TOKEN) for word in word_array]
            if len(word_array) <= seq_len:
                pad_size = seq_len - len(word_array)
                for i in range(pad_size):
                    word_array.append(word2idx.get('<pad>'))
                X[ind, :] = np.array(word_array).astype(np.int32)
            else:
                X[ind, :] = np.array(word_array).astype(np.int32)[0:seq_len]


    return X

def translationAugmentYandex(X, y, key, export_path, begin = 0):
    n_negative = np.sum(y)
    X_ret = []
    n_sentences_processed = 0
    for ind in range(X.shape[0]):
        if y[ind] == 1:
            if n_sentences_processed >= begin:
                try:
                    en = re.sub(r'[\\][tnrfv]', ' ', X[ind])
                    en = re.sub(r'[\\]xa0', ' ', en)
                    en = re.sub(r'[\\]+[^\w]', '', en)
                    en = re.sub(r'[^\w \'\\]', lambda x: " " + x.group(0) + " ", en)
                    # en = re.sub(r'[.,?!:;=-]', lambda x: " " + x.group(0) + " ", en)
                    en = re.sub(r' +', ' ', en)
                    en = re.sub(r'[^\w \'<>]', '', en)

                    translator = YandexTranslate(key)
                    de_en = translator.translate(
                        translator.translate(en, lang = "en-zh")["text"][0],
                        lang = "zh-en"
                    )["text"][0]
                    es_en = translator.translate(
                        translator.translate(en, lang = "en-ko")["text"][0],
                        lang = "ko-en"
                    )["text"][0]
                    print(X[ind])
                    print(de_en)
                    print(es_en)
                    X_ret.append(de_en)
                    X_ret.append(es_en)
                    n_sentences_processed += 1

                except YandexTranslateException:
                    y_ret = np.ones(shape=((n_sentences_processed - begin) * 2))
                    d = {"Comment": np.array(X_ret), "Insult": y_ret}
                    df = pd.DataFrame(data=d)
                    df.to_csv(path_or_buf=export_path, header = True, index = True)
                    if n_sentences_processed == begin:
                        print("Run out of characters available")
                        return np.array(X_ret)
                    else:
                        str_len = len(export_path)
                        file_name = export_path[0 : str_len - 5]
                        n_csv = str(int(export_path[-5]) + 1)
                        translationAugmentYandex(X, y, key, file_name + n_csv + ".csv", begin = n_sentences_processed)
            else:
                n_sentences_processed += 1

    y_ret = np.ones(shape = ((n_sentences_processed - begin) * 2))
    d = {"Comment": np.array(X_ret), "Insult": y_ret}
    df = pd.DataFrame(data = d)
    df.to_csv(path_or_buf = export_path)

    return np.array(X_ret)
