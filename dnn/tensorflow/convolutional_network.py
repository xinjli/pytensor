import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class Config:

    def __init__(self):
        self.learning_rate = 0.01
        self.training_epochs = 100
        self.display_step = 1
        self.batch_size = 500


class ConvolutionalNetwork:

    def __init__(self, config):
        self.config = config
        self.add_model()

    def conv2d(self, x, w, b, strides=1):
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding="SAME")
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

    def conv_net(self, x, weights, biases, dropout):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)

        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)

        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        fc1 = tf.nn.dropout(fc1, dropout)

        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def add_model(self):

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        self.weights = {
            'wc1' : tf.Variable(tf.random_normal([5, 5, 1, 32])),
            'wc2' : tf.Variable(tf.random_normal([5, 5, 32, 64])),
            'wd1' : tf.Variable(tf.random_normal([7*7*64, 1024])),
            'out' : tf.Variable(tf.random_normal([1024, 10]))
        }

        self.biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([10]))
        }

        self.pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)


    def train(self):

        for epoch in range(self.config.training_epochs):
            total_batch = int(mnist.train.num_examples/self.config.batch_size)
            avg_cost = 0.0

            for i in range(total_batch):

                batch_x, batch_y = mnist.train.next_batch(self.config.batch_size)

                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: 1.0})

                avg_cost += c / total_batch
                print("---- Batch: ", '%04d' % (i), "cost=","{:.9f}".format(c))


            if(epoch % self.config.display_step == 0):
                print("Epoch: ", '%04d' % (epoch +1), "cost=","{:.9f}".format(avg_cost))

            print("Finished")
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            print("Accuracy: ", self.sess.run([accuracy], feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels, self.keep_prob: 1.0}))


if __name__ == '__main__':
    config = Config()
    lr = ConvolutionalNetwork(config)
    lr.train()