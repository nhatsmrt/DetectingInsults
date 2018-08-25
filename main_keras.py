import numpy as np
import pandas as pd
from pathlib import Path
import re, os
import nltk
from nltk.corpus import stopwords
from Source import\
    RNNKeras, \
    accuracy, preprocess
from keras.callbacks import EarlyStopping, ModelCheckpoint


## DEFINE PATHS:
path = Path()
d = path.resolve()
data_path = str(d) + "/Data/"
train_path = data_path + "train.csv"
test_path = data_path + "test_with_solutions.csv"
glove_path = os.path.join(data_path, "glove.6B.50d.txt")
weight_save_path = str(d) + "/weights/weights_base.best.hdf5"
# weight_load_path = str(d) + "/weights/model_stacked_birnn.ckpt"
weight_load_path = None
augment_path = data_path + "augmented_data_yandex_0.csv"

RANDOM_STATE = 42

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


## Insert other useful tokens:

### All-uppercase token:
word2idx['<upp>'] = len(embedding_weights)
embedding_weights.append(np.random.rand(EMBEDDING_DIMENSION))

### Number token:
word2idx['<num>'] = len(embedding_weights)
embedding_weights.append(np.random.rand(EMBEDDING_DIMENSION))

### Unknown token:
UNKNOWN_TOKEN = len(embedding_weights)
word2idx['UNK'] = UNKNOWN_TOKEN
embedding_weights.append(np.random.rand(EMBEDDING_DIMENSION))

### Pad token:
word2idx['<pad>'] = len(embedding_weights)
embedding_weights.append(np.zeros(EMBEDDING_DIMENSION))

embedding_weights = np.asarray(embedding_weights, dtype = np.float32)
VOCAB_SIZE = embedding_weights.shape[0]

## READ AND PREPROCESS DATA:
### Read csv files into pd dataframes:
df_train = pd.read_csv(train_path)
df_augmented = pd.read_csv(augment_path)

y = df_train["Insult"].values.reshape(-1, 1)
y_augmented = np.append(
    y,
    df_augmented["Insult"].values.reshape(-1, 1),
    axis = 0)
X_raw = df_train["Comment"].values
X_augmented_raw = np.append(
    X_raw,
    df_augmented["Comment"].values,
    axis = 0
)
seq_len = 500
# X_train = preprocess(X_raw, word2idx, UNKNOWN_TOKEN, seq_len, None)
X_augmented = preprocess(X_augmented_raw, word2idx, UNKNOWN_TOKEN, seq_len, None)

checkpoint = ModelCheckpoint(
    filepath = weight_save_path,
    monitor = 'val_loss', verbose = 1,
    save_best_only = True)
early = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
callbacks_list = [checkpoint, early]
batch_size = 2
epochs = 10

model = RNNKeras(
    embeddingMatrix = embedding_weights,
    embed_size = EMBEDDING_DIMENSION,
    max_features = VOCAB_SIZE,
    maxlen = seq_len
)
model.fit(
    X_augmented, y_augmented,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1,
    callbacks = callbacks_list
)