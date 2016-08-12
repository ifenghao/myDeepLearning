# coding:utf-8
__author__ = 'zfh'

import numpy as np
import theano
import theano.tensor as T
import utils


# def nin(X, param, shape):
#     w1, w2 = param
#     map0 = []
#     for i in xrange(shape[0]):
#         map1 = []
#         for j in xrange(shape[1]):
#             Xj = X[:, j, :, :].dimshuffle(0, 'x', 1, 2)
#             w1ij = w1[i, j, :, :, :].dimshuffle(0, 'x', 1, 2)
#             w2ij = w2[i, j, :].dimshuffle('x', 0, 'x', 'x')
#             tmp = T.nnet.conv2d(Xj, w1ij, border_mode='half')
#             tmp = T.nnet.relu(tmp, alpha=0)
#             map1.append(T.nnet.conv2d(tmp, w2ij, border_mode='valid'))
#         map0.append(T.nnet.relu(T.sum(map1, axis=0), alpha=0))
#     return T.concatenate(map0, axis=1)
#
#
# X = T.tensor4('X')
# w1=utils.weightInit((64,32,16,3,3))
# w2=utils.weightInit((64,32,16))
# w3=utils.weightInit((128,64,16,3,3))
# w4=utils.weightInit((128,64,16))
#
# layer=nin(X,(w1,w2),(64,32))
# layer=nin(layer,(w3,w4),(128,64))
#
# f=theano.function([X],layer,allow_input_downcast=True)
#
# a=np.arange(32000).reshape((10,32,10,10))
# print f(a).shape


# def nin(X, w, shape):
#     map0 = []
#     for i in xrange(10):
#         map1 = []
#         for j in xrange(shape):
#             Xi = X[i, :, :, :].dimshuffle('x',0,  1, 2)
#             wj = w[j, :, :, :].dimshuffle('x',0,  1, 2)
#             map1.append(T.nnet.conv2d(Xi, wj, border_mode='half'))
#         map0.append(T.concatenate(map1,axis=1))
#     return T.concatenate(map0, axis=0)
#
#
# X = T.tensor4('X')
# w1=utils.weightInit((64,32,3,3),'')
# w2=utils.weightInit((128,64,3,3),'')
#
# layer=nin(X,w1,64)
# layer=nin(layer,w2,128)
#
# f=theano.function([X],layer,allow_input_downcast=True)
#
# a=np.arange(32000).reshape((10,32,10,10))
# print f(a).shape


# def nin(X, w):
#     map0 = []
#     for i in xrange(10):
#         Xi = X[i, :, :, :].dimshuffle('x',0,  1, 2)
#         map0.append(T.nnet.conv2d(Xi, w, border_mode='half'))
#     return T.concatenate(map0, axis=0)
#
#
# X = T.tensor4('X')
# w1=utils.weightInit((64,32,3,3),'')
# w2=utils.weightInit((128,64,3,3),'')
#
# layer=nin(X,w1)
# layer=nin(layer,w2)
#
# f=theano.function([X],layer,allow_input_downcast=True)
#
# a=np.arange(32000).reshape((10,32,10,10))
# print f(a).shape


# def nin(X, w):
#     map0 = []
#     span=T.arange(10)
#     splits=T.ones_like(span,dtype='int')
#     for i in T.split(span,splits,10):
#         Xi = X[i, :, :, :]
#         map0.append(T.nnet.conv2d(Xi, w, border_mode='half'))
#     return T.concatenate(map0, axis=0)
#
#
# X = T.tensor4('X')
# w1=utils.weightInit((64,32,3,3),'')
# w2=utils.weightInit((128,64,3,3),'')
#
# layer=nin(X,w1)
# layer=nin(layer,w2)
#
# f=theano.function([X],layer,allow_input_downcast=True)
#
# a=np.arange(32000).reshape((10,32,10,10))
# print f(a).shape


# def nin(X, w, shape):
#     span1=T.arange(10)
#     splits1=T.ones_like(span1,dtype='int')
#     span2=T.arange(shape)
#     splits2=T.ones_like(span2,dtype='int')
#     map0 = []
#     for i in T.split(span1,splits1,10):
#         map1 = []
#         for j in T.split(span2,splits2,shape):
#             Xi = X[i, :, :, :]
#             wj = w[j, :, :, :]
#             map1.append(T.nnet.conv2d(Xi, wj, border_mode='half'))
#         map0.append(T.concatenate(map1,axis=1))
#     return T.concatenate(map0, axis=0)
#
#
# X = T.tensor4('X')
# w1=utils.weightInit((64,32,3,3),'')
# w2=utils.weightInit((128,64,3,3),'')
#
# layer=nin(X,w1,64)
# layer=nin(layer,w2,128)
#
# f=theano.function([X],layer,allow_input_downcast=True)
#
# a=np.arange(32000).reshape((10,32,10,10))
# print f(a).shape


def conv1(X, w):
    X = X.dimshuffle(0, 'x', 1, 2, 3)
    w = w.dimshuffle(0, 'x', 1, 2, 3)
    index0 = T.arange(X.shape[0], dtype='int32')
    index0 = T.repeat(index0, w.shape[0], axis=0)
    index1 = T.arange(w.shape[0], dtype='int32')
    index1 = T.tile(index1, X.shape[0])
    results, updates = theano.scan(fn=lambda i, j, X, w: T.nnet.conv2d(X[i], w[j], border_mode='half'),
                                   sequences=[index0, index1],
                                   outputs_info=None,
                                   non_sequences=[X, w])  # results是一个tensorvaraiable，当作numpy.array处理，返回的结果按照第一维度拼接
    return T.squeeze(results).reshape((X.shape[0], w.shape[0], results.shape[-2], results.shape[-1]))


def conv2(X, w):
    X = X.dimshuffle(0, 1, 'x', 2, 3)
    w = w.dimshuffle(0, 1, 'x', 'x', 2, 3)
    index0 = T.arange(w.shape[0], dtype='int32')
    index0 = T.repeat(index0, w.shape[1], axis=0)
    index1 = T.arange(w.shape[1], dtype='int32')
    index1 = T.tile(index1, w.shape[0])
    results, updates = theano.scan(
        fn=lambda i, j, X, w: T.nnet.conv2d(X[:, j, :, :, :], w[i, j, :, :, :, :], border_mode='half'),
        sequences=[index0, index1],
        outputs_info=None,
        non_sequences=[X, w])  # results是一个tensorvaraiable，当作numpy.array处理，返回的结果按照第一维度拼接
    results = results.squeeze()
    metaShape = results.shape[-3], results.shape[-2], results.shape[-1]
    results = results.reshape((w.shape[0], w.shape[1]) + metaShape)
    sumed = T.sum(results, axis=1)
    return T.transpose(sumed, axes=(1, 0, 2, 3))


def conv3(X, w):
    X = X.dimshuffle(0, 1, 'x', 2, 3)
    w = w.dimshuffle(0, 1, 'x', 2, 3)
    index0 = T.arange(w.shape[1], dtype='int32')
    results, updates = theano.scan(
        fn=lambda i, X, w: T.nnet.conv2d(X[:, i, :, :, :], w[:, i, :, :, :], border_mode='half'),
        sequences=[index0],
        outputs_info=None,
        non_sequences=[X, w])  # results是一个tensorvaraiable，当作numpy.array处理，返回的结果按照第一维度拼接
    return T.sum(results, axis=0)


X = T.tensor4('X')
shape1 = (64, 32, 3, 3)
shape2 = (128, 64, 3, 3)
w1 = theano.shared(utils.floatX(np.arange(np.prod(shape1)).reshape(shape1)), borrow=True)
w2 = theano.shared(utils.floatX(np.arange(np.prod(shape2)).reshape(shape2)), borrow=True)

layerc = T.nnet.conv2d(X, w1, border_mode='half')
layerc = T.nnet.conv2d(layerc, w2, border_mode='half')
f0 = theano.function([X], layerc, allow_input_downcast=True)

layer1 = conv1(X, w1)
layer1 = conv1(layer1, w2)
f1 = theano.function([X], layer1, allow_input_downcast=True)

layer2 = conv2(X, w1)
layer2 = conv2(layer2, w2)
f2 = theano.function([X], layer2, allow_input_downcast=True)

layer3 = conv3(X, w1)
layer3 = conv3(layer3, w2)
f3 = theano.function([X], layer3, allow_input_downcast=True)

a = np.arange(32000).reshape((10, 32, 10, 10))
r0 = f0(a)
r1 = f1(a)
r2 = f2(a)
r3 = f3(a)

print np.allclose(r0, r1), np.allclose(r0, r2), np.allclose(r0, r3)
