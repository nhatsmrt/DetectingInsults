import numpy as np
import pandas as pd
from pathlib import Path
import re, os
import nltk
from nltk.corpus import stopwords
from Source import\
    Seq2Seq, \
    accuracy, preprocess
from sklearn.model_selection import train_test_split


# DEFINE PATHS:
path = Path()
d = path.resolve()
data_path = str(d) + "/Data/"
pretrain_path = data_path + "imdb_master.csv"
glove_path = os.path.join(data_path, "glove.6B.50d.txt")
weight_save_path = str(d) + "/weights/RNN_pretrained.ckpt"

RANDOM_STATE = 42

# LOAD GLOVE WEIGHTS
## Adapt from https://damienpontifex.com/2017/10/27/using-pre-trained-glove-embeddings-in-tensorflow/
PAD_TOKEN = 0
embedding_weights = []
word2idx = {'<pad>': PAD_TOKEN }
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
## Insert zero PAD weights at index 0:
embedding_weights.insert(0, np.zeros(EMBEDDING_DIMENSION))

## Insert other useful tokens:\
### Unknown token:
UNKNOWN_TOKEN = len(embedding_weights)
word2idx['UNK'] = UNKNOWN_TOKEN
embedding_weights.append(np.mean(embedding_weights, axis = 0))

### All-uppercase token:
word2idx['<upp>'] = len(embedding_weights)
embedding_weights.append(np.random.rand(EMBEDDING_DIMENSION))

### Number token:
word2idx['<num>'] = len(embedding_weights)
embedding_weights.append(np.random.rand(EMBEDDING_DIMENSION))

embedding_weights = np.asarray(embedding_weights, dtype = np.float32)
VOCAB_SIZE = embedding_weights.shape[0]


# READ AND PREPROCESS DATA:
## Read csv files into pd dataframes:
df_pretrain = pd.read_csv(pretrain_path, encoding = "ISO-8859-1")
X_raw = df_pretrain["review"].values
len_array_pretrain = []
for ind in range(X_raw.shape[0]):
    len_array_pretrain.append(len(X_raw[ind]))
len_array_pretrain = np.array(len_array_pretrain)
ind = np.where(len_array_pretrain <= 200)[0]
X_raw = X_raw[ind]
print(X_raw.shape)

seq_len = 3000
X_raw = preprocess(X_raw, word2idx, UNKNOWN_TOKEN, seq_len, None)
y = X_raw

X_train, X_val, y_train, y_val = train_test_split(
    X_raw,
    y,
    train_size = 0.95,
    random_state = RANDOM_STATE
)

# CREATE MODEL:
model = Seq2Seq(
    keep_prob = 0.5,
    seq_len = seq_len,
    embedding_matrix = embedding_weights,
    vocab_size = VOCAB_SIZE,
    embed_size = EMBEDDING_DIMENSION
)

model.fit(
    X_train,
    X_val,
    num_epochs = 1,
    patience = 5,
    weight_save_path = weight_save_path,
)

