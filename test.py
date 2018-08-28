import numpy as np
import pandas as pd
from pathlib import Path
import re, os
import nltk
from nltk.corpus import stopwords
from Source import\
    SimpleRNN, BiRNN, StackedBiRNN, AttentionalRNN, AttentionalBiRNN, Simple1DConvNet, \
    accuracy, preprocess, write_predictions, write_results
from sklearn.metrics import confusion_matrix, roc_auc_score


OPTIMAL_THRESHOLD = 0.9421564340591431

## DEFINE PATHS:
path = Path()
d = path.resolve()
data_path = str(d) + "/Data/"
train_path = data_path + "train.csv"
test_path = data_path + "test_with_solutions.csv"
glove_path = os.path.join(data_path, "glove.6B.50d.txt")
weight_save_path = str(d) + "/weights/24/SimpleRNN.ckpt"
weight_load_path = str(d) + "/weights/SimpleRNN_augmented.ckpt"
augment_path = data_path + "/augmented_data_yandex_0.csv"
write_test_result_path = str(d) + "/weights/result.txt"
sample_submission_path = str(d) + '/Data/sample_submission_null.csv'
submission_path = str(d) + '/Data/submission.csv'


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
## Insert zero pad weights at index 0:
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
df_train = pd.read_csv(train_path)
df_augmented = pd.read_csv(augment_path)
df_test = pd.read_csv(test_path)

y_train = df_train["Insult"].values.reshape(-1, 1)
y_train_augmented = np.append(
    y_train,
    df_augmented["Insult"].values.reshape(-1, 1),
    axis = 0
)
X_train_raw = df_train["Comment"].values
X_train_augmented_raw = np.append(
    X_train_raw,
    df_augmented["Comment"].values,
    axis = 0
)
seq_len = 3000
X_train = preprocess(X_train_raw, word2idx, UNKNOWN_TOKEN, seq_len, None)

y_test = df_test["Insult"].values.reshape(-1, 1)
X_test_raw = df_test["Comment"].values
X_test = preprocess(X_test_raw, word2idx, UNKNOWN_TOKEN, seq_len, None)

# DEFINE AND TRAIN MODEL:
model = SimpleRNN(
    keep_prob = 0.5,
    seq_len = seq_len,
    embedding_matrix = embedding_weights,
    embed_size = EMBEDDING_DIMENSION
)

model.fit(
    X_train,
    y_train,
    None,
    None,
    num_epochs = 0,
    weight_save_path = weight_save_path,
    weight_load_path = weight_load_path
)

# TEST MODEL PERFORMANCE:
predictions_prob = model.predict(X_test, return_proba = True)

predictions = (predictions_prob > OPTIMAL_THRESHOLD).astype(np.int32)
acc = accuracy(predictions, y_test)
cfs_mat = confusion_matrix(
    y_true = y_test.reshape(y_test.shape[0]),
    y_pred = predictions.reshape(predictions.shape[0])

)
roc_auc = roc_auc_score(
    y_true = y_test.reshape(y_test.shape[0]),
    y_score = predictions_prob.reshape(predictions_prob.shape[0])
)

write_predictions(
    predictions_prob = predictions_prob,
    sample_submission_path = sample_submission_path,
    submission_path = submission_path
)

print()
print("Test Accuracy:")
print(acc)
print("Confusion Matrix:")
print(cfs_mat)
print("ROC AUC:")
print(roc_auc)

## Write test results:
write_results(
    write_test_result_path = write_test_result_path,
    OPTIMAL_THRESHOLD = OPTIMAL_THRESHOLD,
    acc = acc,
    cfs_mat = cfs_mat,
    roc_auc = roc_auc
)