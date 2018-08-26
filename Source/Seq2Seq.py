import numpy as np
import tensorflow as tf
import math
from .SimpleRNN import SimpleRNN

class Seq2Seq:

    def __init__(
            self, n_classes = 2,
            embedding_matrix = None,
            keep_prob = 0.5,
            use_gpu = False,
            seq_len = 200,
            vocab_size = 3000,
            embed_size = 300,
    ):
        tf.set_random_seed(0)
        self._keep_prob = keep_prob
        self._keep_prob_tensor = tf.placeholder(tf.float32, name = "keep_prob_tens")
        self._n_classes = n_classes
        self._seq_len = seq_len
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._embedding_matrix = embedding_matrix

        # with self._g.as_default():
        if use_gpu:
            with tf.device('/device:GPU:0'):
                self.create_network()
        else:
            with tf.device('/device:CPU:0'):
                self.create_network()

            self._init_op = tf.global_variables_initializer()

    # def lstm_layer(self, x, cell):
    #     output, final_states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    #     return output, final_states

    def create_network(self):
        self._X = tf.placeholder(shape=[None, self._seq_len], dtype=tf.int32)
        self._batch_size = tf.placeholder(shape=[], dtype=tf.int32)
        self._is_training = tf.placeholder(tf.bool)

        # Embedding layer:
        if self._embedding_matrix is not None:
            embedding = tf.Variable(initial_value=self._embedding_matrix, name="embedding")
        else:
            np.random.seed(0)
            embedding = tf.Variable(
                initial_value=tf.random_uniform(
                    shape=[self._vocab_size, self._embed_size],
                    minval=-1,
                    maxval=1),
                name="embedding")

        self._X_embed = tf.nn.embedding_lookup(embedding, self._X, name="embed_X")

        # LSTM Layer:
        # self._cell = self.multiple_lstm_cells(n_units = 512, n_layers = 3)
        n_cells = 1
        self._cells_weights_list, self._cell = self.multiple_gru_cells(n_units=128, n_cells = n_cells, name = "gru")
        self._initial_state = self._cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)
        self._encoder_op, self._encoder_final_state = tf.nn.dynamic_rnn(
            cell = self._cell,
            inputs = self._X_embed,
            dtype=tf.float32,
            initial_state=self._initial_state,
            sequence_length=self.length(self._X_embed)
        )
        self._decoder_rnn_op, self._decoder_rnn_final_state = tf.nn.dynamic_rnn(
            cell = self._cell,
            inputs = self._X_embed,
            dtype = tf.float32,
            initial_state = self._encoder_final_state,
            sequence_length = self.length(self._X_embed)
        )
        print(self._decoder_rnn_op)
        self._op = self.decode(self._decoder_rnn_op, n_units = 128)

        # Final feedforward layer and output:
        # self._fc1 = self.feedforward_layer(self._lstm_op_reshape, n_inp = 128, n_op = 256,  name = "fc1")

        self._mean_loss = self.cost(op = self._op, target = self._X_embed)

        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)

        save_dict = dict()
        save_dict['embedding'] = embedding
        for cell_ind in range(n_cells):
            save_dict['gru_' + str(cell_ind)] = self._cells_weights_list[cell_ind]
        self._saver = tf.train.Saver(save_dict)

    def multiple_gru_cells(self, n_units, n_cells, name):
        cells_list = [tf.contrib.rnn.GRUCell(num_units = n_units, name = name + "_" + str(ind)) for ind in range(n_cells)]
        cells_list_dropout = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self._keep_prob_tensor) for cell in cells_list]
        cells_weights_list = [cell.get_weights() for cell in cells_list]
        return cells_weights_list, tf.contrib.rnn.MultiRNNCell(cells_list_dropout)

    def feedforward_layer(self, x, n_inp, n_op, name, final_layer = False):
        W = tf.get_variable(name = "W_" + name, shape = [n_inp, n_op])
        b = tf.get_variable(name = "b_" + name, shape = [n_op])
        z = tf.matmul(x, W) + b

        if final_layer:
            return z
        else:
            a = tf.nn.relu(z)
            # h = tf.layers.batch_normalization(a, training = self._is_training)
            return a

    def decode(self, hidden_states, n_units):
        W = tf.get_variable(name = "W_decode", shape = [n_units, self._embed_size])
        b = tf.get_variable(name = "b_decode", shape = [self._embed_size])

        hidden_states_reshaped = tf.reshape(hidden_states, shape = [-1, n_units])
        op = tf.matmul(hidden_states_reshaped, W) + b

        return tf.reshape(op, shape = [-1, self._seq_len, self._embed_size])

    def fit(
            self, X, X_val,
            num_epochs = 3,
            print_every = 1,
            weight_save_path = None,
            batch_size = 16,
            patience = 3,
    ):
        # with self._g.as_default():

        self._sess = tf.Session()
        self._sess.run(self._init_op)

        iter = 0


        cur_val_loss = 1000
        p = 0

        for e in range(num_epochs):
            print("Epoch " + str(e + 1))
            # state = self._sess.run(self._initial_state, feed_dict = {self._batch_size: batch_size})
            # n_batches = X.shape[0] // batch_size

            train_indicies = np.arange(X.shape[0])
            np.random.shuffle(train_indicies)

            for i in range(int(math.ceil(X.shape[0] // batch_size))):
                start_idx = (i * batch_size) % X.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]
                actual_batch_size = X[idx, :].shape[0]

                state = self._sess.run(self._initial_state, feed_dict={self._batch_size: actual_batch_size})

                feed_dict = {
                    self._X: X[idx, :],
                    self._keep_prob_tensor: self._keep_prob,
                    self._is_training: True,
                    self._batch_size: actual_batch_size,
                    self._initial_state: state
                }

                # print(self._sess.run(self._mean_loss, feed_dict = feed_dict))
                _, loss = self._sess.run(
                    [self._train_step, self._mean_loss],
                    feed_dict = feed_dict)


                if iter % print_every == 0:
                    print("Iteration " + str(iter) + " with loss " + str(loss))

                iter += 1

            # Validation:
            if X_val is not None:
                state = self._sess.run(self._initial_state, feed_dict = {self._batch_size: X_val.shape[0]})
                feed_dict_val = {
                    self._X: X_val,
                    self._keep_prob_tensor: 1.0,
                    self._is_training: False,
                    self._batch_size: X_val.shape[0],
                    self._initial_state: state
                }
                val_loss = self._sess.run(self._mean_loss, feed_dict = feed_dict_val)

                print("Validation loss: " + str(val_loss))


                if val_loss < cur_val_loss:
                    cur_val_loss = val_loss
                    print("Validation loss decreases.")
                    if weight_save_path is not None:
                        save_path = self._saver.save(self._sess, save_path = weight_save_path)
                        print("Model's weights saved at %s" % save_path)
                    p = 0
                else:
                    p += 1
                    if p > patience:
                        return

            else:
                if weight_save_path is not None:
                    save_path = self._saver.save(self._sess, save_path=weight_save_path)
                    print("Model's weights saved at %s" % save_path)

    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def save_weight(self, weight_save_path):
        save_path = self._saver.save(self._sess, save_path=weight_save_path)
        print("Model's weights saved at %s" % save_path)

    def load_weight(self, weight_load_path):
        self._saver.restore(self._sess, save_path=weight_load_path)
        print("Weights loaded successfully.")

    # Adapt from https://danijar.com/variable-sequence-lengths-in-tensorflow/
    def cost(self, op, target):
        # Compute cross entropy for each frame.
        mse = tf.square(target - op)
        mse = tf.reduce_sum(mse, 2)
        mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
        mse *= mask
        # Average over actual sequence lengths.
        mse = tf.reduce_sum(mse, 1)
        mse /= tf.reduce_sum(mask, 1)
        return tf.reduce_mean(mse)






