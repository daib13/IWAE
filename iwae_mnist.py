import tensorflow as tf 
from tensorflow.contrib import layers
import os
import sys
from mnist import MNIST
import data_util as du
import math
import matplotlib.pyplot as plt
import numpy as np


def ip(input, input_dim, output_dim, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [input_dim, output_dim], tf.float32, layers.xavier_initializer(), trainable=True)
        b = tf.get_variable('b', [output_dim], tf.float32, tf.zeros_initializer, trainable=True)
    with tf.name_scope(name):
        net = tf.matmul(input, w) + b
    return net


def ip_tanh(input, input_dim, output_dim, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [input_dim, output_dim], tf.float32, layers.xavier_initializer(), trainable=True)
        b = tf.get_variable('b', [output_dim], tf.float32, tf.zeros_initializer, trainable=True)
    with tf.name_scope(name):
        net = tf.matmul(input, w) + b
        activate = tf.nn.tanh(net, 'activate')
    return activate


def ip_sigmoid(input, input_dim, output_dim, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [input_dim, output_dim], tf.float32, layers.xavier_initializer(), trainable=True)
        b = tf.get_variable('b', [output_dim], tf.float32, tf.zeros_initializer, trainable=True)
    with tf.name_scope(name):
        net = tf.matmul(input, w) + b
        activate = tf.nn.sigmoid(net, 'activate')
    return activate


class IWAE_MNIST:
    def __init__(self, batch_size, input_dim, hidden_dim, num_hidden, latent_dim, output_dim, k, reuse):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.latent_dim = latent_dim
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.k = k
        self.reuse = reuse
        self.output_dim = output_dim
        self.protector = 1e-20

    def __create__placeholder(self):
        with tf.name_scope("data"):
            self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim], name='x')
        with tf.name_scope("label"):
            self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dim], name='y_gt')

    def __create__encoder(self):
        with tf.name_scope("binarize"):
            self.x_ref = tf.random_uniform([self.batch_size, self.input_dim, 1], 0, 1, name='x_ref')
            self.x_reshape = tf.reshape(self.x, [self.batch_size, self.input_dim, 1], name='x_reshape')
            self.x_concat = tf.concat([self.x_ref, self.x_reshape], 2)
            self.x_binary = tf.cast(tf.arg_max(self.x_concat, 2), tf.float32, 'x_binary')

        with tf.name_scope("encoder"):
            self.encoder = []
            self.previous_tensor = self.x_binary
            self.previous_dim = self.input_dim
            for i_hidden in range(self.num_hidden):
                scope_name = 'encoder' + str(i_hidden)
                self.encoder.append(ip_tanh(self.previous_tensor, self.previous_dim, self.hidden_dim,
                                            scope_name, reuse=self.reuse))
                self.previous_tensor = self.encoder[i_hidden]
                self.previous_dim = self.hidden_dim
            self.mu_z = ip_tanh(self.previous_tensor, self.previous_dim, self.latent_dim, 'mu_z', reuse=self.reuse)
            self.logsd_z = ip_tanh(self.previous_tensor, self.previous_dim, self.latent_dim, 'logsd_z', reuse=self.reuse)
            self.sd_z = tf.exp(self.logsd_z, 'sd_z')

    def __create_decoder(self):
        with tf.name_scope("sampling"):
            self.mu_z_tile = tf.tile(tf.reshape(self.mu_z, [self.batch_size, 1, self.latent_dim]), [1, self.k, 1])
            self.sd_z_tile = tf.tile(tf.reshape(self.sd_z, [self.batch_size, 1, self.latent_dim]), [1, self.k, 1])
            self.noise = tf.random_normal([self.batch_size, self.k, self.latent_dim], name='noise')
            self.z = tf.multiply(self.sd_z_tile, self.noise) + self.mu_z_tile
            self.z_reshape = tf.reshape(self.z, [self.batch_size*self.k, self.latent_dim])

        with tf.name_scope("decoder"):
            self.decoder = []
            self.previous_tensor = self.z_reshape
            self.previous_dim = self.latent_dim
            for i_hidden in range(self.num_hidden):
                scope_name = 'decoder' + str(i_hidden)
                self.decoder.append(ip_tanh(self.previous_tensor, self.previous_dim, self.hidden_dim,
                                            scope_name, reuse=self.reuse))
                self.previous_tensor = self.decoder[i_hidden]
                self.previous_dim = self.hidden_dim
            self.x_hat = ip_sigmoid(self.previous_tensor, self.previous_dim, self.input_dim, 'x_hat', reuse=self.reuse)
            self.x_hat_reshape = tf.reshape(self.x_hat, [self.batch_size, self.k, self.input_dim])

    def __create_classifier(self):
        with tf.name_scope("classifier"):
            self.feature = tf.stop_gradient(self.mu_z, name='feature')
        self.output_logit = ip(self.feature, self.latent_dim, self.output_dim, 'classifier', reuse=self.reuse)

    def __create_loss(self):
        with tf.name_scope("vae_loss"):
            self.logit_z = tf.log(self.sd_z_tile + self.protector) + tf.square(self.noise) / 2 - tf.square(self.z) / 2
            self.x_tile = tf.tile(tf.reshape(self.x, [self.batch_size, 1, self.input_dim]), [1, self.k, 1])
            self.logit_x = tf.multiply(self.x_tile, tf.log(self.x_hat_reshape + self.protector)) \
                           + tf.multiply(1 - self.x_tile, tf.log(1 - self.x_hat_reshape + self.protector))
            self.logit = tf.reduce_sum(self.logit_z, 2) + tf.reduce_sum(self.logit_x, 2)
            self.logit_max = tf.reduce_max(self.logit, 1)
            self.logit_max_tile = tf.tile(tf.reshape(self.logit_max, [self.batch_size, 1]), [1, self.k])
            self.res = tf.reduce_sum(tf.exp(self.logit - self.logit_max_tile), 1)
            self.vae_loss = -tf.reduce_sum(self.logit_max + tf.log(self.res), name='loss') / self.batch_size
        with tf.name_scope("classifier_loss"):
            self.classifier_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logit,
                                                                                         labels=self.y),
                                                 name='loss') / self.batch_size

    def __create_summary(self):
        with tf.name_scope("summary"):
            self.summary_vae = tf.summary.scalar('vae_loss', self.vae_loss)
            self.summary_classifier = tf.summary.scalar('classifier_loss', self.classifier_loss)

    def build_graph(self):
        self.__create__placeholder()
        self.__create__encoder()
        self.__create_decoder()
        self.__create_classifier()
        self.__create_loss()
        self.__create_summary()


class IWAE_OPTIMIZER:
    def __init__(self, model):
        with tf.name_scope('optimizer'):
            self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                    epsilon=0.0001).minimize(model.vae_loss,
                                                                             global_step=model.global_step)


def train_model(model, optimizer, x, y):
    saver = tf.train.Saver()
    if not os.path.exists('./model'):
        os.mkdir('./model')

    with tf.Session() as sess:
        # initialize
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # train autoencoder
        writer = tf.summary.FileWriter('graph', sess.graph)

        iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))
        iteration = 0
        for stage in range(5):
            epoch_in_this_stage = 3**stage
            lr = 0.001 * 10**(-float(stage)/7.0)
            for epoch in range(epoch_in_this_stage):
                x, y = du.shuffle_data(x, y)
                total_loss = 0
                for i in range(iteration_per_epoch):
                    start_idx = model.batch_size * i
                    end_idx = start_idx + model.batch_size
                    feed_dict = {model.x: x[start_idx:end_idx, :], model.y: y[start_idx:end_idx, :],
                                 optimizer.lr: lr}
                    batch_loss, _, summary = sess.run([model.vae_loss, optimizer.optimizer, model.summary_vae],
                                                      feed_dict=feed_dict)
                    total_loss += batch_loss
                    iteration += 1
                    if i == iteration_per_epoch - 1:
                        writer.add_summary(summary, iteration)
                total_loss /= iteration_per_epoch
                print('Stage = {0}, epoch = {1}, loss = {2}.'.format(stage, epoch, total_loss))
                if epoch == epoch_in_this_stage - 1:
                    saver.save(sess, 'model/VAE' + str(stage))


def test_model(model, x, y):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        iteration_per_epoch = int(math.floor(x.shape[0] / model.batch_size))
        nll = 0
        for index in range(iteration_per_epoch):
            start_idx = model.batch_size * index
            end_idx = start_idx + model.batch_size
            feed_dict = {model.x: x[start_idx:end_idx, :], model.y: y[start_idx:end_idx, :]}
            batch_nll = sess.run(model.vae_loss, feed_dict=feed_dict)
            nll += batch_nll
        nll /= iteration_per_epoch
        return nll


def main(k):
    images_train, labels_train = du.load_mnist_data('training')
    images_test, labels_test = du.load_mnist_data('testing')

    model = IWAE_MNIST(20, 784, 200, 2, 50, 10, k, None)
    model.build_graph()
    optimizer = IWAE_OPTIMIZER(model)

    model_test = IWAE_MNIST(20, 784, 200, 2, 50, 10, 5000, True)
    model_test.build_graph()

#    orig_stdout = sys.stdout
#    f = open('log.txt', 'w')
#    sys.stdout = f

    train_model(model, optimizer, images_train, labels_train)
    nll = test_model(model_test, images_test, labels_test)
    print('Test NLL = {0}.'.format(nll))

#    sys.stdout = orig_stdout
#    f.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
#    k = int(sys.argv[1])
#    alpha = float(sys.argv[2])
#    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
#    main(k, alpha)
    main(1)
