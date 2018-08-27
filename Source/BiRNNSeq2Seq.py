import numpy as np
import tensorflow as tf
import math
from .Seq2Seq import Seq2Seq

class BiRNNSeq2Seq(Seq2Seq):

    def __init__(
            self, n_classes = 2,
            embedding_matrix = None,
            keep_prob = 0.5,
            use_gpu = False,
            seq_len = 200,
            vocab_size = 3000,
            embed_size = 300,
    ):
        super().__init__(n_classes, embedding_matrix, keep_prob, use_gpu, seq_len, vocab_size, embed_size)

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

        self._X_embed = tf.stop_gradient(tf.nn.embedding_lookup(embedding, self._X, name = "embed_X"))

        # LSTM Layer:
        # self._cell = self.multiple_lstm_cells(n_units = 512, n_layers = 3)
        self._cell_fw = self.multiple_gru_cells(n_units = 128, n_cells = 1, name = "gru_fw")
        self._cell_bw = self.multiple_gru_cells(n_units = 128, n_cells = 1, name = "gru_bw")
        self._initial_state_fw = self._cell_fw.zero_state(batch_size = self._batch_size, dtype = tf.float32)
        self._initial_state_bw = self._cell_fw.zero_state(batch_size = self._batch_size, dtype = tf.float32)
        self._lstm_op, self._final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = self._cell_fw,
            cell_bw = self._cell_bw,
            inputs = self._X_embed,
            dtype = tf.float32,
            initial_state_fw = self._initial_state_fw,
            initial_state_bw = self._initial_state_bw,
            sequence_length = self.length(self._X_embed)
        )
        self._save_list = tf.trainable_variables()[1:]
        self._lstm_op_concat = tf.concat(self._lstm_op, 2)
        self._lstm_op_reshape = tf.reshape(tf.squeeze(self._lstm_op_concat[:, -1]), (self._batch_size, -1))
        self._final_states_concat = tf.concat(self._final_states, 2)
        self._final_state_fw = self._final_states[0]
        self._final_state_bw = self._final_states[1]



        # Final feedforward layer and output:
        # self._fc1 = self.feedforward_layer(self._lstm_op_reshape, n_inp = 256, n_op = 512,  name = "fc1")
        self._decoder_rnn_op, self._decoder_rnn_final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw = self._cell_fw,
            cell_bw=self._cell_bw,
            inputs = self._X_embed,
            dtype = tf.float32,
            initial_state_fw =self._final_state_fw,
            initial_state_bw =self._final_state_bw,
            sequence_length = self.length(self._X_embed)
        )
        self._decoder_op_concat = tf.concat(self._lstm_op, 2)
        self._op = self.decode(self._decoder_op_concat, n_units = 256)


        # Final feedforward layer and output:
        # self._fc1 = self.feedforward_layer(self._lstm_op_reshape, n_inp = 128, n_op = 256,  name = "fc1")

        self._mean_loss = self.cost(op = self._op, target = self._X_embed)


        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)
        self._init_op = tf.global_variables_initializer()


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
        self._saver = tf.train.Saver(self._save_list)

        iter = 0


        cur_val_loss = 1000
        p = 0

        for e in range(num_epochs):
            print("Epoch " + str(e + 1))
            # state_fw = self._sess.run(self._initial_state_fw, feed_dict = {self._batch_size: batch_size})
            # state_bw = self._sess.run(self._initial_state_bw, feed_dict = {self._batch_size: batch_size})

            # n_batches = X.shape[0] // batch_size

            train_indicies = np.arange(X.shape[0])
            np.random.shuffle(train_indicies)

            for i in range(int(math.ceil(X.shape[0] // batch_size))):
                start_idx = (i * batch_size) % X.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]
                actual_batch_size = X[idx, :].shape[0]

                state_fw = self._sess.run(self._initial_state_fw, feed_dict={self._batch_size: actual_batch_size})
                state_bw = self._sess.run(self._initial_state_bw, feed_dict={self._batch_size: actual_batch_size})

                feed_dict = {
                    self._X: X[idx, :],
                    self._keep_prob_tensor: self._keep_prob,
                    self._batch_size: actual_batch_size,
                    self._is_training: True,
                    self._initial_state_fw: state_fw,
                    self._initial_state_bw: state_bw
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
                state_fw = self._sess.run(self._initial_state_fw, feed_dict={self._batch_size: X_val.shape[0]})
                state_bw = self._sess.run(self._initial_state_bw, feed_dict={self._batch_size: X_val.shape[0]})

                feed_dict_val = {
                    self._X: X_val,
                    self._keep_prob_tensor: 1.0,
                    self._is_training: False,
                    self._batch_size: X_val.shape[0],
                    self._initial_state_fw: state_fw,
                    self._initial_state_bw: state_bw
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


