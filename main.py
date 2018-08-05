import numpy as np
import pandas as pd
from pathlib import Path
import re, os
import nltk
from nltk.corpus import stopwords
from Source import\
    SimpleRNN, BiRNN, StackedBiRNN, RNNKeras, AttentionalRNN, AttentionalBiRNN, \
    accuracy, preprocess
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


## DEFINE PATHS:
path = Path()
d = path.resolve()
data_path = str(d) + "/Data/"
train_path = data_path + "train.csv"
test_path = data_path + "test_with_solutions.csv"
glove_path = os.path.join(data_path, "glove.6B.50d.txt")
weight_save_path = str(d) + "/weights/model_attentional_rnn.ckpt"
# weight_load_path = str(d) + "/weights/model_stacked_birnn.ckpt"
weight_load_path = None
augment_path = data_path + "/augmented_data_yandex_0.csv"

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
    axis = 0
)
X_raw = df_train["Comment"].values
X_augmented_raw = np.append(
    X_raw,
    df_augmented["Comment"].values,
    axis = 0
)
seq_len = 500
X_raw = preprocess(X_augmented_raw, word2idx, UNKNOWN_TOKEN, seq_len, None)
X_train, X_val, y_train, y_val = train_test_split(
    X_raw,
    y_augmented,
    train_size = 0.9,
    random_state = RANDOM_STATE
)

## DEFINE AND TRAIN MODEL:
model = AttentionalBiRNN(
    keep_prob = 0.5,
    seq_len = seq_len,
    embedding_matrix = embedding_weights,
    embed_size = EMBEDDING_DIMENSION
)

model.fit(
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs = 10,
    weight_save_path = weight_save_path,
    weight_load_path = None
)


## VALIDATE MODEL PERFORMANCE:
predictions = model.predict(X_val)
print("Final Validation Accuracy:")
print(accuracy(predictions, y_val))
print(confusion_matrix(
    y_true = y_val.reshape(y_val.shape[0]),
    y_pred = predictions.reshape(predictions.shape[0]))
)
print(roc_auc_score(
    y_true = y_val.reshape(y_val.shape[0]),
    y_score = predictions.reshape(predictions.shape[0]))
)

