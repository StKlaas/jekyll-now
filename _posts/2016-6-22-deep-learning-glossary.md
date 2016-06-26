---
layout: post
title: Deep Learning Glossary
---

[TOC]

由于正在进行深度学习的研究,主要用的语言是python. 在实际写程序的过程中, 经常会遇到一些技巧性的东西,特此下来来并且不断更新, 如果有任何疑问, 麻烦在下方留言或者联系邮箱 strikerklaas@gmail.com.

---
# Dropout
Dropout  is a recently introduced regularization method that has been very successful with feed-forward neural networks.
it is a regularization technique for Neural Networks that prevents overfitting. It prevents neurons from co-adapting by randomly setting a fraction of them to 0 at each training iteration. 
Here is the Python code
```python
from theano.tensor.shared_randomstreams import RandomStreams

def dropout_layer(state_prev, dropout_percent):                
    state=T.switch(trng.binomial(size=state_prev.shape,          p=dropout_percent), state_prev, 0)
    return state
```
Reference:
[Recurrent Neural network regularization](http://arxiv.org/abs/1409.2329)


---
# F
## float32 (theano)
The default floating point data type is _float64_, however, data must be tranferred to _float32_ to store in the GPU.

- convert to _float32_
```
epilson = np.float32(0.01)
```
- use _shared_ statement
``` 
import theano
import theano.tensor as T
w = theano.shared((np.random.randn(input_dimension,output_dimension).astype('float32'), name='w')
```
---
# L
## Loss Function
### zero-one loss
The objective of training a classifier is to minimize the number of errors on unseen examples. The zero-one loss is a very simple classifier returns a value of `0` if the prediction is true and `1` if the prediction in wrong!
We'd like to maximize the probability of seeing $Y=k$ if $x$ and parameter $\theta$ are given. In this glossary, $f$ is defined as: $$f(x) = argmax_k P(Y = k|x,\theta) $$
In python:
```python
# T.neq returns the result of logical inequality(x!=y)
zero_one_loss = T.sum(T.neq(T.argmax(p_y_given_x),y))
```
### negative log likelihood loss
Generally, the zero-one loss is not differentiable and require large computation. In practice, the negative log likelihood loss is widely used and proven to be powerful!
$$ NLL(\theta, D) = - \sum_{i=0}^{|D|} log P(Y = y^{(i)}|x^{(i)},\theta)$$
In python:
```
# This function is a little trivial, please be serious and looks what happen inside. 
# T.log(p_y_given_x) is the log value of p_y,noted as LP 
# y.shape[0] returns the number of our training samples.
# LP[T.arange(y.shape[0],y] is a vector contains LP[0,y[0]]...LP[n-1,y[n-1]] and finally sum up all of them!

NLL = -T.sum(T.log(p_y_given_x)[T.arange(y.shape[0]),y])
```
It's just like the tip that we generate a one-hot vector, but at this time, we select an exact element(y[n]) from row-n. 

---
# M
## MNIST dataset
The MNIST dataset is a universally-used dataset for digit recognition, its characters can be summed up as the following:

1. train set:` 50,000`, validation set:`10,000`,test set:`10,000`
2. `28 x 28` pixels (input `x`of each training example is represented as a 1-dimensional array whose length is `784` while output label `y` is simply a scalar ranging from `0` to `9`.

Now, we begin with opening the dataset in Python and try to optimize it to be used for GPU acceleration.
```python
import cPickle, gzip, numpy, theano
import theano.tensor as T
# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# Next, store the data into GPU memory
def share_dataset(data_xy):
    # use theano shared value form
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    '''
    Can also use the following syntax, it also works!
    shared_x = theano.shared(data_x.astype('float32'))
    shared_y = theano.shared(data_y.astype('float32'))
    '''
    # Since 'Y' should be intergers, not floats, we cast it
    return shared_x, T.cast(shared_y, 'int32')
    
# Now try it!
test_set_x, test_set_y = share_dataset(test_set)
valid_set_x, valid_set_y = share_dataset(valid_set)
train_set_x, train_set_y = share_dataset(train_set)

```
It is very common to use batch gradient descent(see later) rather than use the whole dataset, for  it is  computational heavilly!
```python
# Use batch_zize of 500 and select the third batch.
batch_size = 500
data = train_set_x[2 * batch_size: 3 * batch_size]
label = train_set_y[2 * batch_size: 3 * batch_size]
```
---

# O
## one-hot vector
one-hot vector 在自然语言中处理非常重要, 常作为神经网络的输入, 有indexing的效果. 那么,实际情况中如何建立这样一个矩阵呢. 先考虑小的数据集. 比如有数据标记为两类0,1
one-hot vector is a term in NLP, as its name indicates, it is a vector where only one element is 1 and the others are 0s. Suppose that we have a vocabulary consists of 4000 words for text generation, there should exist 4000 unique one-hot vector for each word. For different tasks, there are different ways to initialize the vectors.

- classification
Suppose that there are only 2 classes: 0 and 1. The two one-hot vectors should be ```[1,0],[0,1]```. suppose that we have six learning samples but they are store in an array like ```[0,1,0,1,1,0]```, so, we produce an eye matrix first and let the array selects which vector they belong to form a matrix includes all samples.
``` Python
>>> import numpy as np
>>> x = np.eye(2) # Two types of vectors
>>> y = np.array([0,1,0,1,1,0]) # classes
>>> x
array([[ 1.,  0.],
       [ 0.,  1.]])
>>> y
array([0, 1, 0, 1, 1, 0])
>>> x[y] # By indexing, we generate a matrix for learning
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 1.,  0.],
       [ 0.,  1.],
       [ 0.,  1.],
       [ 1.,  0.]])
```
---

# R
## Regularization
### L1 and L2 regularization
$L1$ and $L2$ regularization involves adding an extra term to the loss function to prevent the problem of overfitting. Formally, 
$$ E(\theta,D) = NLL(\theta,D) + \lambda ||\theta||_p^P$$
where
$$ ||\theta||_p = (\sum_{j=0}^{|\theta|}|\theta_j|^p)^{1/p}$$
which is the $L_p$ norm of $\theta$.
In python:
```python
L1 = T.sum(abs(param))
L2 = T.sum(param**2)
loss = NLL + lambda_1 * L1 + lambda_2 * L2
```
### early-stopping
Early-stopping combats overfitting by monitoring the model's performance on a validation set. If the model's performance ceases to improve sufficiently on the validation set, or even degrades with further optimization, we give up further optimization.

---

# S
## Stochastic Gradient Descent
### gradient descent
An ordinary gradient descent refer to the training method in which we repeatedly make small steps downward on an error surface defined by a loss function of some parameters.
Pseudocode:
```pyton
while True:
	loss = f(params)
	d_loss_wrt_params = ... #compute gradient
	params -= learning_rate * d_loss_wrt_params
	if <stopping_condition is met>:
		return params
``` 
### stochastic gradient descent
Stochastic gradient descent works just like gradient descent but proceeds quickly by estimating the gradient from just a few samples at a time instead of the entire dataset.
Pseudocode:
```python
for (x_i,y_i) in training_set:
    loss = f(params, x_i, y_i)
    d_loo_wrt_params = ...
    params -= learning_rate * d_loss_wrt_params
    if <stopping_condition> is met>:
        return params
```
### minibatch SGD
More importantly, it is recommended to use minibatches rather than use just one training example at a time. The technique reduces variance in the estimate of the gradient.
Pseudocode:
```
for (x_batch, y_batch) in train_batches:
    loss = f(params, x_batch, y_batch)
    d_loss_wrt_params = ... 
    params -= learning_rate * d_loss_wrt_params
    if <stopping_condition is met>:
        return params
```
**Note**:  minibatch size is dependent of our model,dataset and hardware, ranging from 1 to several hundreds. In `deeplearning.net` tutorial,  it is set to `20`.
**Warning**: If we are training for a fixed number of epochs, the minibatch size becomes important !

In `theano`, the general form of gradient descent is as follows:
```
# Symbolic description of parameters and fucntions!
d_loss_wrt_params = T.grad(loss,params)
updates = [(params, params - learning_rate * d_loss_wrt_params)]
MSGD = theano.function([x_batch, y_batch],loss, update = updates)
for(x_batch, y_batch) in train_batchs:
	print ("current loss is ",MSGD(x_batch,y_batch))
	if stopping_condition_is_met:
		return params
```


---

# Reference
[1] http://deeplearning.net/tutorial/
