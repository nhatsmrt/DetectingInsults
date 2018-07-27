import numpy as np
import tensorflow as tf
import math
from .SimpleRNN import SimpleRNN

class AttentionalRNN(SimpleRNN):

    def __init__(
            self, n_classes = 1,
            embedding_matrix = None,
            keep_prob = 0.5,
            use_gpu = False,
            seq_len = 200,
            n_words = 3000,
            embed_size = 300,
    ):
        self._keep_prob = keep_prob
        self._keep_prob_tensor = tf.placeholder(tf.float32, name = "keep_prob_tens")
        self._n_classes = n_classes
        self._seq_len = seq_len
        self._n_words = n_words
        self._embed_size = embed_size
        self._embedding_matrix = embedding_matrix

        self._g = tf.Graph()

        # with self._g.as_default():
        if use_gpu:
            with tf.device('/device:GPU:0'):
                self.create_network()
        else:
            with tf.device('/device:CPU:0'):
                self.create_network()

            self._saver = tf.train.Saver()
            self._init_op = tf.global_variables_initializer()





    def create_network(self):
        self._X = tf.placeholder(shape = [None, self._seq_len], dtype = tf.int32)
        self._batch_size = tf.placeholder(shape = [], dtype = tf.int32)
        self._is_training = tf.placeholder(tf.bool)


        # Embedding layer:
        if self._embedding_matrix is not None:
            embedding = tf.Variable(initial_value = self._embedding_matrix, name = "embedding")
        else:
            embedding = tf.Variable(
                initial_value = tf.random_uniform(
                    shape = [self._n_words, self._embed_size],
                    minval = -1,
                    maxval = 1),
                name="embedding")

        self._X_embed = tf.nn.embedding_lookup(embedding, self._X, name = "embed_X")

        # GRU Layer:
        self._cell = self.multiple_gru_cells(n_units = 128, n_cells = 1)
        self._initial_state = self._cell.zero_state(batch_size = self._batch_size, dtype = tf.float32)
        self._lstm_op, self._final_state = tf.nn.dynamic_rnn(
            cell = self._cell,
            inputs = self._X_embed,
            dtype = tf.float32,
            initial_state = self._initial_state)

        # Final feedforward layer and output:
        self._attention = self.attention_module(
            self._lstm_op,
            n_units = 128,
            op_size = 64,
            name = "attention"
        )
        self._fc = self.feedforward_layer(self._attention, n_inp = 128, n_op = self._n_classes, final_layer = True, name = "fc")
        self._op = tf.nn.sigmoid(self._fc)

        self._y = tf.placeholder(name = "y", shape = [None, 1], dtype = tf.float32)
        self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self._y, logits = self._fc))
        self._correct_pred = tf.equal(tf.round(self._op), self._y)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))


        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)

    def attention_module(self, hidden_states, name, n_units, op_size):
        W = tf.get_variable(name = "W_" + name, shape = [n_units, op_size])
        b = tf.get_variable(name = "b_" + name, shape = [op_size])

        hidden_states_reshaped = tf.reshape(hidden_states, shape = [-1, n_units])
        rep = tf.tanh(tf.matmul(hidden_states_reshaped, W) + b)

        context = tf.get_variable(name = "cont_" + name, shape = [op_size, 1])
        score = tf.nn.softmax(
            tf.reshape(tf.matmul(rep, context), (-1, self._seq_len)),
            axis = -1
        )

        final_vec = tf.reduce_sum(
            hidden_states * tf.expand_dims(score, axis = -1),
            axis = 1
        )
        return final_vec


