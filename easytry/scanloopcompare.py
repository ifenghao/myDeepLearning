import numpy as np
import theano
import theano.tensor as T
from theano import scan
from theano.tensor.nnet import conv2d, relu, categorical_crossentropy

import load
from utils import basicUtils, gradient, initial, preprocess


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


def gap(X, param):
    wgap = param
    layer = conv2d(X, wgap, border_mode='valid')
    layer = relu(layer, alpha=0)
    layer = T.mean(layer, axis=(2, 3))
    return layer


def metaOp(i, j, X, w1, w2):
    hiddens = conv2d(X[:, j, :, :, :], w1[i, j, :, :, :, :], border_mode='valid')
    hiddens = relu(hiddens, alpha=0)
    outputs = conv2d(hiddens, w2[i, j, :, :, :, :], border_mode='valid')
    return relu(outputs)


def nin1(X, param):
    w1, w2 = param
    X = X.dimshuffle(0, 1, 'x', 2, 3)
    w1 = w1.dimshuffle(0, 1, 2, 'x', 3, 4)
    w2 = w2.dimshuffle(0, 1, 'x', 2, 'x', 'x')
    indexi = T.arange(w1.shape[0], dtype='int32')
    indexi = T.repeat(indexi, w1.shape[1], axis=0)
    indexj = T.arange(w1.shape[1], dtype='int32')
    indexj = T.tile(indexj, w1.shape[0])
    results, updates = scan(fn=metaOp,
                            sequences=[indexi, indexj],
                            outputs_info=None,
                            non_sequences=[X, w1, w2],
                            strict=True)
    metaShape = results.shape[-4], results.shape[-2], results.shape[-1]
    reshaped = results.reshape((w1.shape[0], w2.shape[1]) + metaShape)
    sumed = T.sum(reshaped, axis=1)
    permuted = T.transpose(sumed, axes=(1, 0, 2, 3))
    return permuted


def nin2(X, param, shape):
    w1, w2 = param
    map0 = []
    for i in xrange(shape[0]):
        map1 = []
        for j in xrange(shape[1]):
            Xj = X[:, j, :, :].dimshuffle(0, 'x', 1, 2)
            w1ij = w1[i, j, :, :, :].dimshuffle(0, 'x', 1, 2)
            w2ij = w2[i, j, :].dimshuffle('x', 0, 'x', 'x')
            tmp = conv2d(Xj, w1ij, border_mode='valid')
            tmp = relu(tmp, alpha=0)
            map1.append(conv2d(tmp, w2ij, border_mode='valid'))
        map0.append(relu(T.sum(map1, axis=0), alpha=0))
    return T.concatenate(map0, axis=1)


X = T.tensor4('X')
Y = T.matrix('Y')
shape11 = (4, 3, 5, 3, 3)
shape12 = (4, 3, 5)
w11 = theano.shared(basicUtils.floatX(np.arange(np.prod(shape11)).reshape(shape11)), borrow=True)
w12 = theano.shared(basicUtils.floatX(np.arange(np.prod(shape12)).reshape(shape12)), borrow=True)
# shapegap = (10, 4, 1, 1)
# wgap = theano.shared(utils.floatX(np.arange(np.prod(shapegap)).reshape(shapegap)), borrow=True)

shape21 = (5, 4, 5, 3, 3)
shape22 = (5, 4, 5)
w21 = theano.shared(basicUtils.floatX(np.arange(np.prod(shape21)).reshape(shape21)), borrow=True)
w22 = theano.shared(basicUtils.floatX(np.arange(np.prod(shape22)).reshape(shape22)), borrow=True)
shapegap = (10, 5, 1, 1)
wgap = theano.shared(basicUtils.floatX(np.arange(np.prod(shapegap)).reshape(shapegap)), borrow=True)

layer1 = nin1(X, [w11, w12])
layer1 = nin1(layer1, [w21, w22])
layer1 = gap(layer1, wgap)
YDropProb1 = softmax(layer1)
trNeqs = basicUtils.neqs(YDropProb1, Y)
trCrossEntropy = categorical_crossentropy(YDropProb1, Y)
trCost1 = T.mean(trCrossEntropy)
updates1 = basicUtils.sgd(trCost1, [w11, w12, wgap], 0.001)
f1 = theano.function([X, Y], trCost1, updates=updates1, allow_input_downcast=True)

layer2 = nin2(X, [w11, w12], shape11)
layer2 = nin2(layer2, [w21, w22], shape21)
layer2 = gap(layer2, wgap)
YDropProb2 = softmax(layer2)
trNeqs = basicUtils.neqs(YDropProb2, Y)
trCrossEntropy = categorical_crossentropy(YDropProb2, Y)
trCost2 = T.mean(trCrossEntropy)
updates2 = basicUtils.sgd(trCost2, [w11, w12, wgap], 0.001)
f2 = theano.function([X, Y], trCost2, updates=updates2, allow_input_downcast=True)

x = np.random.randint(0, 100, (500, 3, 10, 10))
y = load.one_hot(np.random.randint(0, 10, (500,)), 10)
for start, end in zip(range(0, 500, 10), range(10, 500, 10)):
    r1 = f1(x[start:end], y[start:end])
    r2 = f2(x[start:end], y[start:end])
    print r1, r2, np.allclose(r1, r2)
