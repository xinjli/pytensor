from data.image.mnist.mnist import *
from network.model.two_layer_net import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_loss_lst = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

learning_rate = 0.1

network = TwoLayerNet(input_size = 784, hidden_size=50, output_size=10)

for i in range(iters_num):

    print("now iterating "+str(i))

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_lst.append(loss)

