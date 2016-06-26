---
layout: post
title: Deep Learning Glossary
---

# Dropout
Dropout  is a recently introduced regularization method that has been very successful with feed-forward neural networks.
it is a regularization technique for Neural Networks that prevents overfitting. It prevents neurons from co-adapting by randomly setting a fraction of them to 0 at each training iteration. 
Here is the Python code
```python
from theano.tensor.shared_randomstreams import RandomStreams

def dropout_layer(state_prev, dropout_percent):                
    state=T.switch(trng.binomial(size=state_prev.shape, p=dropout_percent), state_prev, 0)
    return state
```
Reference:
[Recurrent Neural network regularization](http://arxiv.org/abs/1409.2329)

# Reference
[1] http://deeplearning.net/tutorial/
