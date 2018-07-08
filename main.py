import numpy as np
import pandas as pd
from pathlib import Path
import re, os
import nltk
from Source import SimpleLSTM, accuracy

## DEFINE PATHS:
path = Path()
d = path.resolve()
data_path = str(d) + "/Data/"
train_path = data_path + "train.csv"
test_path = data_path + "test_with_solutions.csv"
glove_path = os.path.join(data_path, "glove.6B.50d.txt")

# LOAD GLOVE WEIGHTS
## Adapt from https://damienpontifex.com/2017/10/27/using-pre-trained-glove-embeddings-in-tensorflow/
PAD_TOKEN = 0
embedding_weights = []
word2idx = { 'PAD': PAD_TOKEN }
with open(glove_path, 'r') as file:
    for ind, line in enumerate(file):
        values = line.split()
        word = values[0]
        word_weights = np.asarray(values[1:], dtype = np.float32)
        word2idx[word] = ind + 1
        embedding_weights.append(word_weights)

        if ind + 1 == 40000:
            break

EMBEDDING_DIMENSION = len(embedding_weights[0])
## Insert random PAD weights at index 0:
embedding_weights.insert(0, np.random.rand(EMBEDDING_DIMENSION))
## Insert unknown:
UNKNOWN_TOKEN = len(embedding_weights)
word2idx['UNK'] = UNKNOWN_TOKEN
embedding_weights.append(np.random.rand(EMBEDDING_DIMENSION))

embedding_weights = np.asarray(embedding_weights, dtype = np.float32)
VOCAB_SIZE = embedding_weights.shape[0]

## READ AND PREPROCESS DATA:
df_train = pd.read_csv(train_path)
y_train = df_train["Insult"].values.reshape(-1, 1)
X_train_raw = df_train["Comment"].values
seq_len = 4200
X_train = np.zeros((len(X_train_raw), seq_len), dtype = np.int32)
for ind in range(X_train_raw.shape[0]):
    word_array = re.sub(r'[^\w]', ' ', X_train_raw[ind]).split()
    # for word in word_array:
    #     word = nltk.word_tokenize(word)
    word_array = [word2idx.get(word, UNKNOWN_TOKEN) for word in word_array]
    X_train[ind, -len(word_array):] = np.array(word_array).astype(np.int32)

df_test = pd.read_csv(test_path)
y_test = df_test["Insult"].values.reshape(-1, 1)
X_test_raw = df_test["Comment"].values
X_test = np.zeros((len(X_test_raw), seq_len), dtype = np.int32)
for ind in range(X_test_raw.shape[0]):
    word_array = re.sub(r'[^\w]', ' ', X_test_raw[ind]).split()
    # for word in word_array:
    #     word = nltk.word_tokenize(word)
    word_array = [word2idx.get(word, UNKNOWN_TOKEN) for word in word_array]
    X_test[ind, -len(word_array):] = np.array(word_array).astype(np.int32)


## DEFINE AND TRAIN MODEL:
model = SimpleLSTM(
    seq_len = seq_len,
    embedding_matrix = embedding_weights,
    embed_size = EMBEDDING_DIMENSION)

model.fit(X_train, y_train, num_epochs = 1)

## TEST MODEL PERFORMANCE:
predictions = model.predict(X_test)
print("Test Accuracy:")
print(accuracy(predictions, y_test))
