import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import math
from .SimpleRNN import SimpleRNN

class BiRNN(SimpleRNN):

    def __init__(
            self, n_classes = 2,
            embedding_matrix = None,
            pretrained_weight_path=None,
            keep_prob = 0.8,
            use_gpu = False,
            seq_len = 200,
            vocab_size = 3000,
            embed_size = 300,
    ):
        super().__init__(n_classes, embedding_matrix, pretrained_weight_path, keep_prob, use_gpu, seq_len, vocab_size, embed_size)

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
                    shape = [self._vocab_size, self._embed_size],
                    minval = -1,
                    maxval = 1),
                name="embedding")

        self._X_embed = tf.nn.embedding_lookup(embedding, self._X, name = "embed_X")

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
        self._pretrain_weights_list = tf.trainable_variables()[1:]
        self._lstm_op_concat = tf.concat(self._lstm_op, 2)
        self._lstm_op_reshape = tf.reshape(tf.squeeze(self._lstm_op_concat[:, -1]), (self._batch_size, -1))
        self._final_states_concat = tf.concat(self._final_states, 2)
        self._final_states_reshape = tf.reshape(self._final_states_concat, shape = [-1, 256])


        # Final feedforward layer and output:
        # self._fc1 = self.feedforward_layer(self._lstm_op_reshape, n_inp = 256, n_op = 512,  name = "fc1")
        self._fc = self.feedforward_layer(self._final_states_reshape, n_inp = 256, n_op = 1, final_layer = True, name = "fc")
        self._op = tf.nn.sigmoid(self._fc)

        self._y = tf.placeholder(name = "y", shape = [None, 1], dtype = tf.float32)
        # self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self._y, logits = self._fc))
        self._mean_loss = self.cost_pairwise(self._op, self._y)
        self._correct_pred = tf.equal(tf.round(self._op), self._y)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))


        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)

    def fit(
            self, X, y, X_val, y_val,
            num_epochs = 3,
            print_every = 1,
            weight_save_path = None,
            weight_load_path = None,
            batch_size = 16,
            patience = 3,
            val_metric = 'acc'
    ):
        # with self._g.as_default():

        self._sess = tf.Session()
        self._sess.run(self._init_op)

        iter = 0

        if weight_load_path is not None:
            self._saver.restore(self._sess, save_path = weight_load_path)
            print("Weights loaded successfully.")
        elif self._pretrained_weight_path is not None:
            self._cell_saver = tf.train.Saver(self._pretrain_weights_list)
            self._cell_saver.restore(self._sess, save_path = self._pretrained_weight_path)
            print("Pretrained weight loaded successfully.")


        cur_val_loss = 1000
        cur_val_acc = 0
        cur_val_roc_auc = 0
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
                actual_batch_size = y[idx].shape[0]

                state_fw = self._sess.run(self._initial_state_fw, feed_dict={self._batch_size: actual_batch_size})
                state_bw = self._sess.run(self._initial_state_bw, feed_dict={self._batch_size: actual_batch_size})

                feed_dict = {
                    self._X: X[idx, :],
                    self._y: y[idx],
                    self._keep_prob_tensor: self._keep_prob,
                    self._batch_size: actual_batch_size,
                    self._is_training: True,
                    self._initial_state_fw: state_fw,
                    self._initial_state_bw: state_bw

                }

                _, loss, acc, states = self._sess.run(
                    [self._train_step, self._mean_loss, self._accuracy, self._final_states],
                    feed_dict = feed_dict)

                # print(self._sess.run(self._final_states_concat, feed_dict = feed_dict).shape)
                state_fw = states[0]
                state_bw = states[1]

                predictions = self.predict(X[idx, :], return_proba = True, verbose = False)
                try:
                    roc_auc = roc_auc_score(
                        y_true = y[idx].reshape(y[idx].shape[0]),
                        y_score = predictions.reshape(predictions.shape[0])
                    )
                except ValueError:
                    roc_auc = 0

                if iter % print_every == 0:
                    print("Iteration " + str(iter) + " with loss " + str(loss) + " and accuracy " + str(acc) + " and roc-auc " + str(roc_auc))

                iter += 1

            # Validation:
            if X_val is not None:
                state_fw = self._sess.run(self._initial_state_fw, feed_dict = {self._batch_size: X_val.shape[0]})
                state_bw = self._sess.run(self._initial_state_bw, feed_dict = {self._batch_size: X_val.shape[0]})
                feed_dict_val = {
                    self._X: X_val,
                    self._y: y_val,
                    self._keep_prob_tensor: 1.0,
                    self._is_training: False,
                    self._batch_size: X_val.shape[0],
                    self._initial_state_fw: state_fw,
                    self._initial_state_bw: state_bw
                }
                val_loss, val_acc = self._sess.run([self._mean_loss, self._accuracy], feed_dict = feed_dict_val)

                print("Validation loss: " + str(val_loss))
                print("Validation accuracy: " + str(val_acc))

                if val_metric == 'acc':
                    if val_acc > cur_val_acc:
                        cur_val_acc = val_acc
                        print("Validation accuracy increases.")
                        if weight_save_path is not None:
                            save_path = self._saver.save(self._sess, save_path = weight_save_path)
                            print("Model's weights saved at %s" % save_path)
                        p = 0
                    else:
                        p += 1
                        if p > patience:
                            return
                elif val_metric == 'loss':
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
                elif val_metric == 'roc-auc':
                    predictions = self.predict(X_val, batch_size = X_val.shape[0], return_proba = True, verbose = False)
                    val_roc_auc = roc_auc_score(
                        y_true = y_val.reshape(y_val.shape[0]),
                        y_score = predictions.reshape(predictions.shape[0])
                    )
                    if val_roc_auc > cur_val_roc_auc:
                        cur_val_roc_auc = val_roc_auc
                        print("Validation ROC-AUC: " + str(val_roc_auc))
                        print("Validation ROC-AUC increases.")
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

    def predict(self, X, return_proba = False, threshold = 0.5, batch_size = 32, verbose = True, use_pairwise_loss = False):
        if batch_size is None:
            batch_size = X.shape[0]

        state_fw = self._sess.run(self._initial_state_fw, feed_dict={self._batch_size: batch_size})
        state_bw = self._sess.run(self._initial_state_bw, feed_dict={self._batch_size: batch_size})
        train_indicies = np.arange(X.shape[0])

        # Find the score of the imaginary class (equals to logit of the threshold) (for pairwise loss)
        imaginary_class_score = np.log(threshold/(1 - threshold))
        true_prob = tf.nn.sigmoid(self._fc - imaginary_class_score)
        prob = np.zeros((X.shape[0], 1), dtype = np.float32)


        if verbose:
            print("Begin Predicting:")
        for i in range(int(math.ceil(X.shape[0] // batch_size))):
            if verbose:
                print("Batch " + str(i + 1))
            start_idx = (i * batch_size) % X.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]
            actual_batch_size = X[idx, :].shape[0]

            feed_dict = {
                self._X: X[idx, :],
                self._keep_prob_tensor: 1.0,
                self._batch_size: actual_batch_size,
                self._initial_state_fw: state_fw,
                self._initial_state_bw: state_bw,
                self._is_training: False
            }
            if use_pairwise_loss:
                prob[idx, :] = self._sess.run(true_prob, feed_dict=feed_dict)
            else:
                prob[idx, :] = self._sess.run(self._op, feed_dict=feed_dict)

        if return_proba:
            return prob

        if use_pairwise_loss:
            return (prob > 0.5).astype(np.int32)

        return (prob > threshold).astype(np.int32)



