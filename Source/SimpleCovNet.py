import tensorflow as tf
import numpy as np
import scipy.misc
import skimage
from skimage.transform import rescale, resize, downscale_local_mean
import timeit
import math
import os
import json
from sklearn.datasets import make_classification
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
# Built by Nhat Hoang Pham


class Simple1DConvNet:
    def __init__(
            self, n_classes = 2,
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
            embedding_pretrained = tf.Variable(initial_value = self._embedding_matrix, name = "embedding_pretrained")
        else:
            embedding_pretrained = tf.Variable(
                initial_value = tf.random_uniform(
                    shape = [self._n_words, self._embed_size],
                    minval = -1,
                    maxval = 1),
                    name="embedding_pretrained")
        self._X_embed_pretrained = tf.stop_gradient(tf.nn.embedding_lookup(
            embedding_pretrained,
            self._X,
            name = "X_embed_pretrained"
        ))

        embedding_trained = tf.Variable(
                initial_value = tf.random_uniform(
                    shape = [self._n_words, self._embed_size],
                    minval = -1,
                    maxval = 1),
                    name = "embedding_trained"
        )
        self._X_embed_trained = tf.nn.embedding_lookup(embedding_trained, self._X, name = "X_embed_trained")

        self._X_embed = tf.concat([self._X_embed_trained, self._X_embed_pretrained], axis = -1)


        self._conv_layer_1 = self.conv_1d_layer(
            self._X_embed,
            name = "conv_layer_1",
            stride = self._embed_size,
            filter_width = 3,
            inp_channel = self._embed_size * 2,
            op_channel = 512
        )
        self._conv_layer_1_pooled = self.max_over_time_pooling(self._conv_layer_1)

        # self._conv_layer_2 = self.conv_1d_layer(
        #     self._conv_layer_1_pooled,
        #     name = "conv_layer_2",
        #     stride = self._embed_size,
        #     filter_width = 3 * self._embed_size,
        #     inp_channel = 16,
        #     op_channel = 32
        # )
        # self._conv_layer_2_pooled = tf.layers.max_pooling1d(
        #     self._conv_layer_2,
        #     pool_size = 2,
        #     strides = 2 * self._embed_size
        # )
        #
        # self._conv_layer_3 = self.conv_1d_layer(
        #     self._conv_layer_2_pooled,
        #     name = "conv_layer_3",
        #     stride = self._embed_size,
        #     filter_width = 3 * self._embed_size,
        #     inp_channel = 32,
        #     op_channel = 64
        # )
        # self._conv_layer_3_pooled = tf.layers.max_pooling1d(
        #     self._conv_layer_3,
        #     pool_size = 2,
        #     strides = 2 * self._embed_size
        # )


        self._flat = tf.reshape(self._conv_layer_1_pooled, shape = [-1, 512], name = "flat")
        self._fc1 = self.feed_forward(self._flat, name = "op", inp_channel = 512, op_channel = 1)
        self._op = tf.nn.dropout(self._fc1, keep_prob = self._keep_prob_tensor)

        self._op_prob = tf.nn.sigmoid(self._op, name = "prob")

    def ret_op(self):
        return self._op_prob

# Adapt from Stanford's CS231n Assignment 3
    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False, weight_save_path = None, patience = None, threshold = 0.5):
        # have tensorflow compute accuracy
        correct_prediction = tf.equal(
            tf.cast((self._op_prob > threshold), dtype = tf.float32),
            self._y
        )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define saver:
        saver = tf.train.Saver()

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training
            self._keep_prob_passed = self._keep_prob
        else:
            self._keep_prob_passed = 1.0

        # counter
        iter_cnt = 0
        val_losses = []
        early_stopping_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                if i < int(math.ceil(Xd.shape[0] / batch_size)) - 1:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: training_now,
                                 self._keep_prob_tensor: self._keep_prob_passed}
                    # have tensorflow compute loss and correct predictions
                    # and (if given) perform a training step
                    loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                    # aggregate performance stats
                    losses.append(loss * actual_batch_size)
                    correct += np.sum(corr)

                    # print every now and then
                    if training_now and (iter_cnt % print_every) == 0:
                        # print(session.run(self._op_prob, feed_dict = feed_dict).shape)
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                              .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
                else:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: False,
                                 self._keep_prob_tensor: 1.0}
                    val_loss = session.run(self._mean_loss, feed_dict = feed_dict)
                    print("Validation loss: " + str(val_loss))
                    val_losses.append(val_loss)
                    # if training_now and weight_save_path is not None:
                    if training_now and val_loss <= min(val_losses) and weight_save_path is not None:
                        save_path = saver.save(session, save_path = weight_save_path)
                        print("Model's weights saved at %s" % save_path)
                    if patience is not None:
                        if val_loss > min(val_losses):
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0
                        if early_stopping_cnt > patience:
                            print("Patience exceeded. Finish training")
                            return
                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        print("Finish Training.")
        return total_loss, total_correct


    # Define a max pool layer with size 2x2, stride of 2 and same padding.

    # Predict:
    def predict(self, X, threshold = 0.5, return_prob = False):
        prob = self._sess.run(self._op_prob, feed_dict = {self._X : X, self._is_training : False, self._keep_prob_tensor : 1.0})

        if return_prob:
            return prob

        return (prob > threshold).astype(np.int32)

    # Define layers and modules:
    def conv_1d_layer(
            self, x, name, inp_channel, op_channel, filter_width = 3,
            stride = 1, padding = 'SAME', not_activated = False, dropout = False
    ):
        W_conv = tf.get_variable(
            name = "W_conv_" + name,
            shape = [filter_width, inp_channel, op_channel],
            initializer = tf.keras.initializers.he_normal()
        )
        b_conv = tf.get_variable(
            name = "b_conv" + name,
            initializer = tf.zeros(op_channel)
        )

        z_conv = tf.nn.conv1d(x, W_conv, stride = stride, padding = padding) + b_conv
        if not_activated:
            return z_conv

        a_conv = tf.nn.relu(z_conv)
        if dropout:
            a_conv_dropout = tf.nn.dropout(a_conv, keep_prob = self._keep_prob)
            return a_conv_dropout

        h_conv = tf.layers.batch_normalization(a_conv, training = self._is_training)
        return h_conv


    def max_over_time_pooling(self, x):
        return tf.reduce_max(x, axis = -2)



    def feed_forward(self, x, name, inp_channel, op_channel, op_layer = False):
        W = tf.get_variable("W_" + name, shape = [inp_channel, op_channel], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b_" + name, shape = [op_channel],dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        z = tf.matmul(x, W) + b
        if op_layer:
            # a = tf.nn.sigmoid(z)
            # return a
            return z
        else:
            a = tf.nn.relu(z)
            a_norm = tf.layers.batch_normalization(a, training = self._is_training)
            return a_norm


    # Train:
    def fit(self, X, y, num_epochs = 64, batch_size = 16, weight_save_path = None, weight_load_path = None, patience = None, plot_losses = False):
        self._y = tf.placeholder(tf.float32, shape = [None, 1])
        self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self._op, labels = self._y))
        self._optimizer = tf.train.AdamOptimizer(1e-4)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        self._sess = tf.Session()
        if weight_load_path is not None:
            loader = tf.train.Saver()
            loader.restore(sess = self._sess, save_path = weight_load_path)
            print("Weight loaded successfully")
        else:
            self._sess.run(tf.global_variables_initializer())
        if num_epochs > 0:
            print('Training Characters Classifier for ' + str(num_epochs) +  ' epochs')
            self.run_model(self._sess, self._op_prob, self._mean_loss, X, y, num_epochs, batch_size, 1, self._train_step, weight_save_path = weight_save_path, patience = patience, plot_losses = plot_losses)



    def create_pad(self, n, pad):
        pad_matrix = [[0, 0]]
        for i in range(n-2):
            pad_matrix.append([pad, pad])
        pad_matrix.append([0, 0])
        return tf.constant(pad_matrix)




    def evaluate (self, X, y):
        self.run_model(self._sess, self._op_prob, self._mean_loss, X, y, 1, 16)
