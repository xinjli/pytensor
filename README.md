# PYDL: A Deep Learning framework with pure numpy

PYDL is a deep learning framework implemented with pure numpy.


## Features

The framework is a toy framework implemented by pure numpy.

* It is a dynamic framework which graph can be re-constructed each time when computing forward.
* Users can use it to construct computational graph by connecting operations (as tensorflow and popular frameworks do)
* Auto differentiation is supported, so it is not necessary to implement backward computation by yourself
* Common operations used in NLP and speech is available such as embedding and lstm operations.  

## Operations

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
  
* Speech-related operations
  * CTC (not included yet, prototype is available under network/temp/ctc.py ) 
  
 
## Tutorial

I implemented three models under the tutorial directory to show how to use the framework.
Each model will be introduced as well as the framework itself in my [blog](www.xinjianl.com)


 
 





