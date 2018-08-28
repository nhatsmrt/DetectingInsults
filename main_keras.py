import numpy as np
import pandas as pd
from pathlib import Path
import re, os
import nltk
from nltk.corpus import stopwords
from Source import\
    RNNKeras, \
    accuracy, preprocess, find_threshold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix


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
word2idx = { '<pad>': PAD_TOKEN }
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
X_processed = preprocess(X_raw, word2idx, UNKNOWN_TOKEN, seq_len, None)
X_train, X_val, y_train, y_val = train_test_split(
    X_processed,
    y,
    stratify = y,
    test_size = 0.05,
    random_state=RANDOM_STATE
)

checkpoint = ModelCheckpoint(
    filepath = weight_save_path,
    monitor = 'val_pairwise_loss', verbose = 1,
    mode = 'min',
    save_best_only = True
)
early = EarlyStopping(monitor = "val_pairwise_loss", mode = "min", patience = 3)
callbacks_list = [checkpoint, early]
batch_size = 16
epochs = 100

model = RNNKeras(
    embeddingMatrix = embedding_weights,
    embed_size = EMBEDDING_DIMENSION,
    max_features = VOCAB_SIZE,
    maxlen = seq_len
)
model.fit(
    X_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.1,
    callbacks = callbacks_list,
    validation_data = (X_val, y_val)
)

# FINAL VALIDATION:
model.load_weights(filepath = weight_save_path)
predictions_prob = model.predict(X_val)

optimal_threshold = find_threshold(predictions_prob, y_val)
print("Optimal Threshold: ")
print(optimal_threshold)
predictions = (predictions_prob > optimal_threshold).astype(np.int32)

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

