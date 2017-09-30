import tensorflow as tf 
import os
import sys
import numpy as np
from mnist import MNIST


class IWAE:
    def __init__(self, batch_size, input_dim, hidden_dim, num_hidden, latent_dim, learning_rate, k, alpha):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.k = k
        self.alpha = alpha

    def __create__placeholder(self):
        with tf.name_scope("data"):
            self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim], name='x')

    def __create__encoder(self):
        with tf.name_scope("encoder"):
            self.previous_tensor = self.x
            self.previous_dim = self.input_dim
            self.encoder_w = []
            self.encoder_b = []
            self.encoder = []
            self.encoder_activate = []
            for i_hidden in range(self.num_hidden):
                w_name = 'encoder_w' + str(i_hidden)
                b_name = 'encoder_b' + str(i_hidden)
                self.encoder_w.append(tf.get_variable(w_name, [self.previous_dim, self.hidden_dim], tf.float32,
                    tf.contrib.layers.xavier_initializer(), trainable=True))
                self.encoder_b.append(tf.Variable(tf.zeros([self.hidden_dim]), trainable=True, name=b_name))
                self.encoder.append(tf.matmul(self.previous_tensor, self.encoder_w[i_hidden]) + self.encoder_b[i_hidden])
                self.encoder_activate.append(tf.nn.relu(self.encoder[i_hidden]))
                self.previous_tensor = self.encoder_activate[i_hidden]
                self.previous_dim = self.hidden_dim
            self.mu_z_w = tf.get_variable('mu_z_w', [self.previous_dim, self.latent_dim], tf.float32,
                tf.contrib.layers.xavier_initializer(), trainable=True)
            self.mu_z_b = tf.Variable(tf.zeros([self.latent_dim]), trainable=True, name='mu_z_b')
            self.mu_z = tf.matmul(self.previous_tensor, self.mu_z_w) + self.mu_z_b
            self.logsd_z_w = tf.get_variable('logsd_z_w', [self.previous_dim, self.latent_dim], tf.float32,
                tf.contrib.layers.xavier_initializer(), trainable=True)
            self.logsd_z_b = tf.Variable(tf.zeros([self.latent_dim]), trainable=True, name='logsd_z_b')
            self.logsd_z = tf.matmul(self.previous_tensor, self.logsd_z_w) + self.logsd_z_b
            self.sd_z = tf.exp(self.logsd_z)

    def __create_decoder(self):
        with tf.name_scope("sampling"):
            self.mu_z_tile = tf.tile(tf.reshape(self.mu_z, [self.batch_size, 1, self.latent_dim]), [1, self.k, 1])
            self.sd_z_tile = tf.tile(tf.reshape(self.sd_z, [self.batch_size, 1, self.latent_dim]), [1, self.k, 1])
            self.noise = tf.random_normal([self.batch_size, self.k, self.latent_dim], name='noise')
            self.z = tf.multiply(self.sd_z_tile, self.noise) + self.mu_z_tile
        with tf.name_scope("decoder"):
            self.z_reshape = tf.reshape(self.z, [self.batch_size * self.k, self.latent_dim])
            self.previous_tensor = self.z_reshape
            self.previous_dim = self.latent_dim
            self.decoder_w = []
            self.decoder_b = []
            self.decoder = []
            self.decoder_activate = []
            for i_hidden in range(self.num_hidden):
                w_name = 'decoder_w' + str(i_hidden)
                b_name = 'decoder_b' + str(i_hidden)
                self.decoder_w.append(tf.get_variable(w_name, [self.previous_dim, self.hidden_dim], tf.float32,
                    tf.contrib.layers.xavier_initializer(), trainable=True))
                self.decoder_b.append(tf.Variable(tf.zeros([self.hidden_dim]), trainable=True, name=b_name))
                self.decoder.append(tf.matmul(self.previous_tensor, self.decoder_w[i_hidden]) + self.decoder_b[i_hidden])
                self.decoder_activate.append(tf.nn.relu(self.decoder[i_hidden]))
                self.previous_tensor = self.decoder_activate[i_hidden]
                self.previous_dim = self.hidden_dim
            self.x_hat_w = tf.get_variable('x_hat_w', [self.previous_dim, self.input_dim], tf.float32,
                tf.contrib.layers.xavier_initializer(), trainable=True)
            self.x_hat_b = tf.Variable(tf.zeros([self.input_dim]), trainable=True, name='x_hat_b')
            self.x_hat = tf.matmul(self.previous_tensor, self.x_hat_w) + self.x_hat_b
            self.x_hat_reshape = tf.reshape(self.x_hat, [self.batch_size, self.k, self.input_dim])

    def __create_loss(self):
        with tf.name_scope("loss"):
            self.logit_z = tf.log(self.sd_z_tile + 1e-12) + tf.square(self.noise) / 2 - tf.square(self.z) / 2
            self.x_tile = tf.tile(tf.reshape(self.x, [self.batch_size, 1, self.input_dim]), [1, self.k, 1])
#            self.logit_x = - tf.square(self.x_hat_reshape - self.x_tile) / 2 / self.alpha
#            self.logit_x = - tf.log(tf.square(self.x_hat_reshape - self.x_tile) + self.alpha)
            self.logit_x = - tf.multiply(self.x_hat_reshape, tf.log(self.x_tile + 1e-6)) - tf.multiply(1 - self.x_hat_reshape, tf.log(1 - self.x_tile + 1e-6))
            self.logit = tf.reduce_sum(self.logit_z, 2) + tf.reduce_sum(self.logit_x, 2)
            self.logit_max = tf.reduce_max(self.logit, 1)
            self.logit_max_tile = tf.tile(tf.reshape(self.logit_max, [self.batch_size, 1]), [1, self.k])
            self.res = tf.reduce_sum(tf.exp(self.logit - self.logit_max_tile), 1)
            self.loss = - tf.reduce_sum(self.logit_max + tf.log(self.res)) / self.batch_size

    def __create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def __create_summary(self):
        with tf.name_scope("summary"):
            self.summary = tf.summary.scalar('loss', self.loss)

    def build_graph(self):
        self.__create__placeholder()
        self.__create__encoder()
        self.__create_decoder()
        self.__create_loss()
        self.__create_optimizer()
        self.__create_summary()


def load_mnist_data(flag='training'):
    mndata = MNIST('data/MNIST')
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
            return images, labels
        elif flag == 'testing':
            images, labels = mndata.load_testing()
            return images, labels
        else:
            raise Exception('Flag should be either trainint or testing.')
    except Exception:
        print("Flag error")
        raise



def train_model(model):
    saver = tf.train.Saver()
    if not os.path.exists('./model'):
        os.mkdir('./model')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        total_loss = 0
        writer = tf.summary.FileWriter('graph', sess.graph)
        initial_step = model.global_step.eval()
        finish_step = 200000
        for index in range(initial_step, int(finish_step)):
            batch_loss, _, summary = sess.run([model.loss, model.optimizer, model.summary])
            total_loss += batch_loss
            writer.add_summary(summary, index)

            if (index+1) % 200 == 0:
                print('Iter = {0}, loss = {1}.'.format(index, total_loss / 200))
                total_loss = 0

            if (index+1) % 50000 == 0:
                saver.save(sess, 'model/VAE' + str(index))


def evaluate_wo(model):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        w = model.decoder_w[0].eval(session=sess)
        w_norm = np.sum(np.square(w), 1)
        w_norm_max = np.max(w_norm)
        return np.sum(w_norm > 0.01 * w_norm_max)


def main(k, alpha):
    images, labels = load_mnist_data('trainding')
#    model = IWAE(100, 400, 200, 2, 30, 0.0003, k, alpha)
#    model.build_graph()

#    orig_stdout = sys.stdout
#    f = open('log.txt', 'w')
#    sys.stdout = f

#    train_model(model)
#    num_active = evaluate_wo(model)
#    print('num active = {0}.'.format(num_active))

#    sys.stdout = orig_stdout
#    f.close()


if __name__ == '__main__':
#    k = int(sys.argv[1])
#    alpha = float(sys.argv[2])
#    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[3]
#    main(k, alpha)
    main(1, 1)