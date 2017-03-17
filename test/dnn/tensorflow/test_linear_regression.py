import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

learning_rate = 0.05
training_epochs = 1000
display_step =50

train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

pred = tf.add(tf.mul(w,x), b)
cost = tf.reduce_sum(tf.pow(pred-y, 2)/(2*n_samples))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (train_x, train_y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})

        if((epoch+1) % display_step == 0 ):
            c = sess.run(cost, feed_dict={x: train_X, y: train_Y})
            print("Epoch: ", '%04d' %(epoch+1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(w), 'b=', sess.run(b))

    print("Finished")
    training_cost = sess.run(cost, feed_dict={x: train_X, y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(w), 'b=', sess.run(b))

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(w)*train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()