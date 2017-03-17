import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class Config:

    def __init__(self):
        self.learning_rate = 0.5
        self.training_epochs = 25
        self.display_step = 1
        self.batch_size = 100



class ThreeLayerNetwork:

    def __init__(self, config, dim=[784, 100, 10]):

        self.config = config
        self.dim = dim
        self.add_model()


    def add_model(self):

        self.x = tf.placeholder(tf.float32, [None, self.dim[0]])
        self.y = tf.placeholder(tf.float32, [None, self.dim[2]])

        self.w1 = tf.Variable(np.random.randn(self.dim[0], self.dim[1]), dtype=tf.float32)
        self.h1 = tf.nn.sigmoid(tf.matmul(self.x, self.w1))

        self.w2 = tf.Variable(np.random.randn(self.dim[1], self.dim[2]), dtype=tf.float32)
        self.h2 = tf.matmul(self.h1, self.w2)

        self.pred = tf.nn.softmax(self.h2)
        self.cost = tf.reduce_mean(tf.reduce_sum(-self.y*tf.log(self.pred), reduction_indices=1))

        self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.cost)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)

    def train(self):

        for epoch in range(self.config.training_epochs):
            total_batch = int(mnist.train.num_examples / self.config.batch_size)
            avg_cost = 0.0

            for i in range(total_batch):

                batch_x, batch_y = mnist.train.next_batch(self.config.batch_size)
                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x, self.y: batch_y})
                avg_cost += c / total_batch

            if (epoch % self.config.display_step == 0):
                print("Epoch: ", "%04d" %(epoch+1), "cost=", "{:.9f}".format(avg_cost))

        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        print("Accuracy: ", self.sess.run([accuracy], feed_dict={self.x: mnist.test.images, self.y: mnist.test.labels}))

if __name__ == '__main__':

    config = Config()
    model = ThreeLayerNetwork(config, [784,50,10])
    model.train()