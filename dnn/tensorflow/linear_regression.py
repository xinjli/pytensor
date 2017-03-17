import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

class Config:

    def __init__(self):
        self.learning_rate = 0.01
        self.training_epochs = 1000
        self.display_step = 50

class LinearRegression:

    def __init__(self, config):

        self.config = config
        self.add_model()

    def load_data(self):

        self.train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
        self.train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    def add_model(self):

        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        self.W = tf.Variable(np.random.randn())
        self.b = tf.Variable(np.random.randn())

        self.pred = tf.add(tf.mul(self.x, self.W), self.b)
        self.cost = tf.reduce_sum(tf.pow(self.pred-self.y, 2))

        self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.cost)

    def train(self, train_X, train_Y):

        self.sess = tf.Session()

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.config.training_epochs):
            for (train_x, train_y) in zip(train_X, train_Y):
                self.sess.run(self.optimizer, feed_dict={self.x: train_x, self.y: train_y})

            if(epoch % self.config.display_step == 0):
                c = self.sess.run(self.cost, feed_dict={self.x: train_X, self.y: train_Y})
                print("Epoch ", "%04d" % (epoch), "cost=", "{:.9f}".format(c), 'W=', self.sess.run(self.W), 'b=', self.sess.run(self.b))


    def predict(self, train_X, train_Y):

        pred_Y = self.sess.run(self.pred, feed_dict={self.x: train_X})

        for (train_y, pred_y) in zip(train_Y, pred_Y):
            print(train_y, pred_y)

if __name__ == '__main__':

    config = Config()
    model = LinearRegression(config)

    train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                          7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                          2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    model.train(train_X, train_Y)
    model.predict(train_X, train_Y)