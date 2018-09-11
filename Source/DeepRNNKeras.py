import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM, GRU, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.backend as K


def RNNKeras(embeddingMatrix = None, embed_size = 100, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(GRU(50, return_sequences = True))(x)
    # x = Dropout(0.1)(x)
    x = Bidirectional(GRU(50, return_sequences = True))(x)
    x = Dropout(0.1)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def pairwise_loss(y_true, y_pred):
    op_pair_diff = tf.matrix_band_part(y_pred -
                                       tf.transpose(y_pred), 0, -1)
    mask = tf.matrix_band_part(y_true -
                               tf.transpose(y_true), 0, -1)

    n_terms = tf.reduce_sum(y_true) * (tf.cast(tf.shape(y_true)[0], tf.float32) - tf.reduce_sum(y_true))
    if n_terms == 0:
        return 0

    return - tf.reduce_sum(tf.log(tf.nn.sigmoid(op_pair_diff * mask))) / (n_terms + 1e-8)
