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
        self.batch_size = 100


class LogisticRegression:

    def __init__(self, config):
        self.config = config
        self.add_model()

    def add_model(self):

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])

        self.w = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        self.pred = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)
        self.cost = tf.reduce_mean(tf.reduce_sum(-self.y*tf.log(self.pred), reduction_indices=1))

        self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        self.sess.run(init)

    def train(self):

        for epoch in range(self.config.training_epochs):
            total_batch = int(mnist.train.num_examples/self.config.batch_size)
            avg_cost = 0.0

            for i in range(total_batch):

                batch_x, batch_y = mnist.train.next_batch(self.config.batch_size)

                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})

                avg_cost += c / total_batch

            if(epoch % self.config.display_step == 0):
                print("Epoch: ", '%04d' % (epoch +1), "cost=","{:.9f}".format(avg_cost))

        print("Finished")
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("Accuracy: ", self.sess.run([accuracy], feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels}))


if __name__ == '__main__':
    config = Config()
    lr = LogisticRegression(config)
    lr.train()