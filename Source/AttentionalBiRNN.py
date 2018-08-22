import tensorflow as tf
import numpy as np
from .AttentionalRNN import AttentionalRNN
from .BiRNN import BiRNN

class AttentionalBiRNN(BiRNN, AttentionalRNN):
    def __init__(
            self, n_classes = 2,
            embedding_matrix = None,
            keep_prob = 0.5,
            use_gpu = False,
            seq_len = 200,
            n_words = 3000,
            embed_size = 300
    ):
        super().__init__(
            n_classes,
            embedding_matrix,
            keep_prob,
            use_gpu,
            seq_len,
            n_words,
            embed_size
        )


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

        # LSTM Layer:
        # self._cell = self.multiple_lstm_cells(n_units = 512, n_layers = 3)
        self._cell_fw = self.multiple_gru_cells(n_units = 128, n_cells = 1)
        self._cell_bw = self.multiple_gru_cells(n_units = 128, n_cells = 1)
        self._initial_state_fw = self._cell_fw.zero_state(batch_size = self._batch_size, dtype = tf.float32)
        self._initial_state_bw = self._cell_fw.zero_state(batch_size = self._batch_size, dtype = tf.float32)
        self._lstm_op, self._final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = self._cell_fw,
            cell_bw = self._cell_bw,
            inputs = self._X_embed,
            dtype = tf.float32,
            initial_state_fw = self._initial_state_fw,
            initial_state_bw = self._initial_state_bw)
        self._lstm_op_concat = tf.concat(self._lstm_op, 2)

        # Final feedforward layer and output:
        # self._fc1 = self.feedforward_layer(self._lstm_op_reshape, n_inp = 256, n_op = 512,  name = "fc1")
        self._attention = self.attention_module(
            hidden_states = self._lstm_op_concat,
            name = "attention",
            n_units = 256,
            context_dim = 64
        )
        self._fc = self.feedforward_layer(
            self._attention, n_inp = 256,
            n_op = 1,
            final_layer = True,
            name = "fc"
        )
        self._op = tf.nn.sigmoid(self._fc)

        self._y = tf.placeholder(name = "y", shape = [None, 1], dtype = tf.float32)
        self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self._y, logits = self._fc))
        self._correct_pred = tf.equal(tf.round(self._op), self._y)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))

        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)

