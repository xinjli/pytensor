# pytensor: A Deep Learning Framework with pure numpy

pytensor is a deep learning framework implemented with pure numpy.

## Features

The framework is a toy framework implemented by pure numpy.

* It is a dynamic framework which graph can be re-constructed each time when computing forward.
* Users can use it to construct computational graph by connecting operations (as tensorflow and popular frameworks do)
* Auto differentiation is supported, so it is not necessary to implement backward computation by yourself
* Common operations used in NLP and speech is available such as embedding and lstm operations.  

![Framework](framework.png)

## Install

To install From this repository (recommended):

	git clone https://github.com/xinjli/pytensor
	python setup.py install

This project is also on [pypi](https://pypi.python.org/pypi/pytensor>)

To install from pypi:

	pip install pytensor

## Tutorial

The architecture of pytensor is shown in the previous diagram. 
Its actual implementaion is described in my [blog](http://www.xinjianl.com)

To build a model (*graph*) with pytensor, the basic steps are as follows
* feed input and targets as *variables*
* forward *variables* with *operations* and *loss*
* backward gradients
* optimize

### Variable
*Variable* can be initialized as follows.

```
In [1]: from pytensor import *

In [2]: import numpy as np

In [3]: v = Variable(np.random.random(5), name='hello')

In [4]: print(v)
Variable {name: hello}
- value    : [0.67471201 0.06407413 0.78337818 0.39475087 0.76176572]
- gradient : [0. 0. 0. 0. 0.]
```

### Operation

Following operations are implemented currently or planned to become available

* Arithmetic operations
  * Addition
  * Multiply
  * Matmul

* Nonlinear operations
  * Relu
  * Sigmoid
  * Tanh
  
* Loss operations
  * Softmax CE Loss 
  * Square Loss

* MLP-related operations
  * Affine
  
* NLP-related operations
  * embedding
  * RNN
  * LSTM
  
### Loss
Loss is a special type of operation which should implement *loss* method in addition to the *forward* and *backward*.
Following loss are implemented currently.

* Softmax CE Loss 
* Square Loss
* CTC (not included yet, prototype is available under the ctc branch)

### Graph
To implement a model, we need to inherit the Graph class, and then implement two methods *forward* and *loss*.
*forward* should specify how the model should produce outputs from inputs, and *loss* should implement the logic of loss function.

To train a model, it is highly recommended to initialize *variable*, *operation* and *loss* using *get_operation* and *get_variable* interfaces.
This is to ensure that *variable* and *operation* are registered and managed by *graph* automatically.
Otherwise, their gradients and updates should be handled manually.

Here we show a example of simple MLP model.

```python
from pytensor import *
from pytensor.network.trainer import *
from pytensor.data.digit_dataset import *


class MLP(Graph):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__("mlp")

        # make graph
        self.affine1 = self.get_operation('Affine', {'input_size': input_size, 'hidden_size': hidden_size})
        self.sigmoid = self.get_operation('Sigmoid')
        self.affine2 = self.get_operation('Affine', {'input_size': hidden_size, 'hidden_size': output_size})
        self.softmaxloss = self.get_operation('SoftmaxLoss')

    def forward(self, input_variable):
        affine1_variable = self.affine1.forward(input_variable)
        sigmoid_variable = self.sigmoid.forward(affine1_variable)
        affine2_variable = self.affine2.forward(sigmoid_variable)

        return self.softmaxloss.forward(affine2_variable)

    def loss(self, target_variable):
        return self.softmaxloss.loss(target_variable)


# load digit data for multiclass classification
data_train, data_test, label_train, label_test = digit_dataset()

# create a MLP model with dimensions of 64 input, 30 hidden, 10 output 
model = MLP(64, 30, 10)

# create a trainer
trainer = Trainer(model)

# train 40 epoch
trainer.train(data_train, label_train, data_test, label_test, 40)
```
  
### Customization

Basically, we can add new operations to support more features. 
To implement a new operation, we need to support *forward* and *backward* interfaces.
Be sure to call *register* interface in forward if your backward is not empty.
This is to register the operation so that backward will be called automatically during backpropagation.

For examples, here is the default addition operation.

```python
class Add(Operation):

    def __init__(self, name='add', argument=None, graph=None):
        super(Add, self).__init__(name, argument, graph)


    def forward(self, input_variables):
        """
        Add all variables in the input_variable

        :param input_variables:
        :return:
        """
        self.register(input_variables)

        # value for the output variable
        value = np.zeros_like(self.input_variables[0].value)

        for input_variable in self.input_variables:
            value += input_variable.value

        self.output_variable = Variable(value)

        return self.output_variable

    def backward(self):
        """
        backward grad into each input variable

        :return:
        """

        for input_variable in self.input_variables:
            input_variable.grad += self.output_variable.grad
```

### Tests

You can implement unit test to validate your model and operations are working.

Sample tests are available in pytensor.test. You can run those existing tests with following commands

 
	python -m pytensor.test.test_linear
	python -m pytensor.test.test_mlp
	python -m pytensor.test.test_rnn
	python -m pytensor.test.test_lstm