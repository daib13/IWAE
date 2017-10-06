import tensorflow as tf 
import os
import sys
import numpy as np
from mnist import MNIST


class IWAE_MNIST:
    def __init__(self, batch_size, input_dim, hidden_dim, num_hidden, latent_dim, output_dim, learning_rate, k):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.k = k
        self.output_dim = output_dim
        self.protector = 1e-12

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
                self.encoder_activate.append(tf.nn.tanh(self.encoder[i_hidden]))
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
                self.decoder_activate.append(tf.nn.tanh(self.decoder[i_hidden]))
                self.previous_tensor = self.decoder_activate[i_hidden]
                self.previous_dim = self.hidden_dim
            self.x_hat_w = tf.get_variable('x_hat_w', [self.previous_dim, self.input_dim], tf.float32,
                tf.contrib.layers.xavier_initializer(), trainable=True)
            self.x_hat_b = tf.Variable(tf.zeros([self.input_dim]), trainable=True, name='x_hat_b')
            self.x_hat = tf.nn.sigmoid(tf.matmul(self.previous_tensor, self.x_hat_w) + self.x_hat_b)
            self.x_hat_reshape = tf.reshape(self.x_hat, [self.batch_size, self.k, self.input_dim])

    def __create_loss(self):
        with tf.name_scope("loss"):
            self.logit_z = tf.log(self.sd_z_tile + self.protector) + tf.square(self.noise) / 2 - tf.square(self.z) / 2
            self.x_tile = tf.tile(tf.reshape(self.x, [self.batch_size, 1, self.input_dim]), [1, self.k, 1])
            self.logit_x = tf.multiply(self.x_tile, tf.log(self.x_hat_reshape + self.protector)) + tf.multiply(1 - self.x_tile, tf.log(1 - self.x_hat_reshape + self.protector))
            self.logit = tf.reduce_sum(self.logit_z, 2) + tf.reduce_sum(self.logit_x, 2)
            self.logit_max = tf.reduce_max(self.logit, 1)
            self.logit_max_tile = tf.tile(tf.reshape(self.logit_max, [self.batch_size, 1]), [1, self.k])
            self.res = tf.reduce_sum(tf.exp(self.logit - self.logit_max_tile), 1)
            self.loss = -tf.reduce_sum(self.logit_max + tf.log(self.res)) / self.batch_size

    def __create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def __create_classification(self):
        with tf.name_scope("classifier"):
            self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.output_dim], name='y_gt')
            self.feature = tf.stop_gradient(self.mu_z, name='feature')
            self.classifier_w = tf.get_variable('classifier_w', [self.latent_dim, self.output_dim], tf.float32,
                tf.contrib.layers.xavier_initializer(), trainable=True)
            self.classifier_b = tf.Variable(tf.zeros([self.output_dim]), trainable=True, name='classifier_b')
            self.output_logit = tf.matmul(self.feature, self.classifier_w) + self.classifier_b
            self.classifier_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logit,
                                                                                         labels=self.y),
                                                 name='classifier_loss') / self.batch_size
        with tf.name_scope("evaluate"):
            self.label_gt = tf.argmax(self.y, axis=1, name='label_gt')
            self.label = tf.argmax(self.output_logit, axis=1, name='label')
            self.correct_prediction = tf.equal(self.label_gt, self.label, name='correct_prediction')
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name='accuracy')
        with tf.name_scope("classifier_optimizer"):
            self.classifier_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.classifier_loss)

    def __create_summary(self):
        with tf.name_scope("summary"):
            self.summary_iwae = tf.summary.scalar('loss', self.loss)
            self.summary_classifier = tf.summary.scalar('classifier_loss', self.classifier_loss)

    def build_graph(self):
        self.__create__placeholder()
        self.__create__encoder()
        self.__create_decoder()
        self.__create_loss()
        self.__create_optimizer()
        self.__create_classification()
        self.__create_summary()


def load_mnist_data(flag='training'):
    mndata = MNIST('data/MNIST')
    try:
        if flag == 'training':
            images, labels = mndata.load_training()
        elif flag == 'testing':
            images, labels = mndata.load_testing()
        else:
            raise Exception('Flag should be either training or testing.')
    except Exception:
        print("Flag error")
        raise
    images_array = np.array(images) / 255
    labels_array = np.array(labels)
    one_hot_labels = np.zeros((labels_array.size, labels_array.max() + 1))
    one_hot_labels[np.arange(labels_array.size), labels_array] = 1
    return images_array, one_hot_labels


def train_model(model, x, y, num_epoch):
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
        total_loss = 0
        writer = tf.summary.FileWriter('graph', sess.graph)
        initial_step = model.global_step.eval()
        iteration_per_epoch = x.shape[0] / model.batch_size
        finish_step = num_epoch * iteration_per_epoch
        start_idx = 0
        end_idx = start_idx + model.batch_size
        for index in range(initial_step, int(finish_step)):
            feed_dict = {model.x: x[start_idx:end_idx, :], model.y: y[start_idx:end_idx, :]}
            start_idx += model.batch_size
            end_idx += model.batch_size
            if end_idx >= x.shape[0]:
                start_idx = 0
                end_idx = start_idx + model.batch_size
            batch_loss, _, summary = sess.run([model.loss, model.optimizer, model.summary_iwae], feed_dict=feed_dict)
            total_loss += batch_loss
            writer.add_summary(summary, index)

            if (index+1) % iteration_per_epoch == 0:
                print('Iter = {0}, loss = {1}.'.format(index, total_loss / iteration_per_epoch))
                total_loss = 0

            if (index+1) % iteration_per_epoch == 0:
                saver.save(sess, 'model/VAE' + str(index))

        # train classifier
        total_loss = 0
        finish_step2 = 2 * finish_step
        for index in range(int(finish_step), int(finish_step2)):
            feed_dict = {model.x: x[start_idx:end_idx, :], model.y: y[start_idx:end_idx, :]}
            start_idx += model.batch_size
            end_idx += model.batch_size
            if end_idx >= x.shape[0]:
                start_idx = 0
                end_idx = start_idx + model.batch_size

            batch_loss, _, summary = sess.run([model.classifier_loss, model.classifier_optimizer, model.summary_classifier], feed_dict=feed_dict)
            total_loss += batch_loss
            writer.add_summary(summary, index - finish_step)

            if (index+1) % iteration_per_epoch == 0:
                print('Iter = {0}, loss = {1}.'.format(index, total_loss / iteration_per_epoch))
                total_loss = 0

            if (index+1) % iteration_per_epoch == 0:
                saver.save(sess, 'model/VAE' + str(index))


def test_model(model, x, y):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        total_accuracy = 0
        iteration_per_epoch = int(x.shape[0] / model.batch_size)
        start_idx = 0
        end_idx = start_idx + model.batch_size
        for index in range(iteration_per_epoch):
            feed_dict = {model.x: x[start_idx:end_idx, :], model.y: y[start_idx:end_idx, :]}
            start_idx += model.batch_size
            end_idx = start_idx + model.batch_size
            batch_accuracy = sess.run(model.accuracy, feed_dict=feed_dict)
            total_accuracy += batch_accuracy
        total_accuracy /= iteration_per_epoch
        return total_accuracy


def main(k):
    NUM_EPOCH = 500

    images_train, labels_train = load_mnist_data('training')
    images_test, labels_test = load_mnist_data('testing')

    model = IWAE_MNIST(100, 784, 200, 2, 50, 10, 0.0003, k)
    model.build_graph()

    orig_stdout = sys.stdout
    f = open('log.txt', 'w')
    sys.stdout = f

    train_model(model, images_train, labels_train, NUM_EPOCH)
    accuracy = test_model(model, images_train, labels_train)
    print('Train accuracy = {0}.'.format(accuracy))
    accuracy = test_model(model, images_test, labels_test)
    print('Test accuracy = {0}.'.format(accuracy))
#    num_active = evaluate_wo(model)
#    print('num active = {0}.'.format(num_active))

    sys.stdout = orig_stdout
    f.close()


if __name__ == '__main__':
#    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    k = int(sys.argv[1])
#    alpha = float(sys.argv[2])
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
#    main(k, alpha)
    main(k)
