import numpy as np
import tensorflow as tf
import math

class SimpleLSTM:

    def __init__(
            self, n_classes = 1,
            embedding_matrix = None,
            keep_prob = 0.8,
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




    # def lstm_layer(self, x, cell):
    #     output, final_states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    #     return output, final_states

    def create_network(self):
        self._X = tf.placeholder(shape = [None, self._seq_len], dtype = tf.int32)
        self._batch_size = tf.placeholder(shape = [], dtype = tf.int32)


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
        self._cell = self.multiple_lstm_cells(n_units = 128, n_cells = 1)
        self._initial_state = self._cell.zero_state(batch_size = self._batch_size, dtype = tf.float32)
        self._lstm_op, self._final_state = tf.nn.dynamic_rnn(
            cell = self._cell,
            inputs = self._X_embed,
            dtype = tf.float32,
            initial_state = self._initial_state)
        self._lstm_op_reshape = tf.reshape(tf.squeeze(self._lstm_op[:, -1]), (self._batch_size, -1))

        # Final feedforward layer and output:
        self._fc = self.feedforward_layer(self._lstm_op_reshape, n_inp = 128, n_op = self._n_classes, final_layer = True, name = "fc")
        self._op = tf.nn.sigmoid(self._fc)

        self._y = tf.placeholder(name = "y", shape = [None, 1], dtype = tf.float32)
        self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self._y, logits = self._fc))
        self._correct_pred = tf.equal(tf.round(self._op), self._y)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))


        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)



    def multiple_lstm_cells(self, n_units, n_cells):
        return tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(num_units = n_units),
                output_keep_prob = self._keep_prob_tensor)
            for _ in range(n_cells)]
        )
    # def multiple_lstm_layers(self, cell, x):
    #     output, final_states = tf.nn.dynamic_rnn(cell, x, dtype = tf.float32)
    #     return output, final_states

    def feedforward_layer(self, x, n_inp, n_op, name, final_layer = False):
        W = tf.get_variable(name = "W_" + name, shape = [n_inp, n_op])
        b = tf.get_variable(name = "b_" + name, shape = [n_op])
        z = tf.matmul(x, W) + b

        if final_layer:
            return z
        else:
            a = tf.nn.relu(z)
            return a

    def fit(self, X, y, num_epochs = 3, print_every = 1, weight_save_path = None, weight_load_path = None, batch_size = 16):
        # with self._g.as_default():

        self._sess = tf.Session()
        self._sess.run(self._init_op)

        iter = 0

        if weight_load_path is not None:
            self._saver.restore(self._sess, save_path = weight_load_path)
            print("Weights loaded successfully.")

        for e in range(num_epochs):
            print("Epoch " + str(e))
            state = self._sess.run(self._initial_state, feed_dict = {self._batch_size: batch_size})
            # n_batches = X.shape[0] // batch_size

            train_indicies = np.arange(X.shape[0])
            np.random.shuffle(train_indicies)

            for i in range(int(math.ceil(X.shape[0] // batch_size))):
                start_idx = (i * batch_size) % X.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                actual_batch_size = y[idx].shape[0]
                feed_dict = {
                    self._X: X[idx, :],
                    self._y: y[idx],
                    self._keep_prob_tensor: self._keep_prob,
                    self._batch_size: actual_batch_size,
                    self._initial_state: state
                }

                _, loss, acc, state = self._sess.run(
                    [self._train_step, self._mean_loss, self._accuracy, self._final_state],
                    feed_dict = feed_dict)

                if iter % print_every == 0:
                    print("Iteration " + str(iter) + " with loss " + str(loss) + " and accuracy " + str(acc))

                iter += 1

            if weight_save_path is not None:
                save_path = self._saver.save(self._sess, save_path = weight_save_path)
                print("Model's weights saved at %s" % save_path)

    def predict(self, X, return_proba = False, batch_size = 32):
        if batch_size is None:
            batch_size = X.shape[0]

        state = self._sess.run(self._initial_state, feed_dict = {self._batch_size: batch_size})
        train_indicies = np.arange(X.shape[0])
        prob = np.zeros((X.shape[0], 1), dtype = np.float32)

        print("Begin Predicting:")
        for i in range(int(math.ceil(X.shape[0] // batch_size))):
            print("Batch " + str(i + 1))
            start_idx = (i * batch_size) % X.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]
            actual_batch_size = X[idx, :].shape[0]

            feed_dict = {
                self._X: X[idx, :],
                self._keep_prob_tensor: 1.0,
                self._batch_size: actual_batch_size,
                self._initial_state: state
            }
            prob[idx, :] = self._sess.run(self._op, feed_dict = feed_dict)

        if return_proba:
            return prob

        return np.round(prob)


    # Adapt from Sebastian Rashka's code
    def generate_batch(self, X, y = None, batch_size = 1):
        n_batches = X.shape[0] // batch_size

        return

