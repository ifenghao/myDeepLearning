import numpy as np
import theano
import theano.tensor as T
from theano import function
import utils
import os, cPickle
import pylab
import time


# xx=np.random.randn(5000).reshape((100,2,5,5))
# ff=np.random.randn(72).reshape((4,2,3,3))
#
# X=T.tensor4()
# f=T.tensor4()
#
# z=T.nnet.conv2d(X,f,border_mode='valid')
# f1=function([X,f],z)
#
# map1 = []
# for i in xrange(100):
#     map2 = []
#     for j in xrange(4):
#         map3 = []
#         for k in xrange(2):
#             map3.append(T.nnet.conv2d(X[i][k].dimshuffle('x','x',0,1), f[j][k].dimshuffle('x','x',0,1), border_mode='valid'))
#         map2.append(T.sum(map3, axis=0))
#     map1.append(T.concatenate(map2, axis=1))
# out=T.concatenate(map1, axis=0)
# f2=function([X,f],out)
#
# map1 = []
# for i in xrange(100):
#     map2 = []
#     for j in xrange(4):
#         map3 = []
#         splits=np.ones(2,dtype='int')
#         Xsplit=T.split(X[i],splits,2,axis=0)
#         fsplit=T.split(f[j],splits,2,axis=0)
#         result=map(lambda x,w:T.nnet.conv2d(x.dimshuffle('x',0,1,2),w.dimshuffle('x',0,1,2)),
#                                Xsplit,fsplit)
#         map2.append(T.sum(result, axis=0))
#     map1.append(T.concatenate(map2, axis=1))
# out=T.concatenate(map1, axis=0)
# f3=function([X,f],out)
#
# # print np.allclose(f1(xx,ff),f2(xx,ff))
#
# tic=time.time()
# a=[]
# for i in range(5):
#     a.append(f1(xx,ff).shape)
# print time.time()-tic
#
# tic=time.time()
# b=[]
# for i in range(5):
#     b.append(f2(xx,ff).shape)
# print time.time()-tic
#
# tic=time.time()
# c=[]
# for i in range(5):
#     c.append(f3(xx,ff).shape)
# print time.time()-tic


# def genIndex(shape,fsize):
#     a,b,c,d=shape
#     sub0=np.ones(b*(c-fsize+1)*(d-fsize+1)*fsize*fsize,dtype='int')
#     l0=[]
#     for i in range(a):
#         l0.append(sub0*i)
#     index0=np.hstack(l0)
#
#     sub1=np.ones((c-fsize+1)*(d-fsize+1)*fsize*fsize,dtype='int')
#     l1=[]
#     for i in range(a):
#         for j in range(b):
#             l1.append(sub1*j)
#     index1=np.hstack(l1)
#
#     sub2=np.ones(fsize,dtype='int')
#     l2=[]
#     for i in range(a):
#         for j in range(b):
#             for k in range(c-fsize+1):
#                 for m in range(d-fsize+1):
#                     for n in range(fsize):
#                         l2.append(sub2*(n+k))
#     index2=np.hstack(l2)
#
#     sub3=np.arange(fsize,dtype='int')
#     l33=[]
#     for i in range(fsize):
#         l33.append(sub3)
#     sub33=np.hstack(l33)
#     l3=[]
#     for i in range(a):
#         for j in range(b):
#             for k in range(c-fsize+1):
#                 for m in range(d-fsize+1):
#                     l3.append(sub33+m)
#     index3=np.hstack(l3)
#     return index0,index1,index2,index3

# def genIndex(XShape, filterShape, stride=(1, 1)):
#     nSample, nMap, mapRow, mapCol = XShape
#     filterRow, filterCol = filterShape
#     rowStride, colStride = stride
#     outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
#     block00 = np.arange(nSample, dtype='int')
#     index0 = np.repeat(block00, nMap * outRow * outCol * filterRow * filterCol, axis=0)
#     block10 = np.arange(nMap, dtype='int')
#     block11 = np.repeat(block10, outRow * outCol * filterRow * filterCol, axis=0)
#     index1 = np.tile(block11, nSample)
#     block20 = np.arange(filterRow, dtype='int')
#     block21 = np.repeat(block20, filterCol, axis=0)
#     block22 = np.tile(block21, outCol * outRow)
#     arange2 = np.arange(outRow, dtype='int')
#     added2 = np.repeat(arange2, filterRow * filterCol * outCol, axis=0)
#     index2 = np.tile(block22 + added2, nSample * nMap)
#     block30 = np.arange(filterCol, dtype='int')
#     block31 = np.tile(block30, filterRow * outCol)
#     arange3 = np.arange(outCol, dtype='int')
#     added3 = np.repeat(arange3, filterRow * filterCol)
#     index3 = np.tile(block31 + added3, outRow * nSample * nMap)
#     return index0, index1, index2, index3
#
#
# def convdot2d(X, filter, index, border_mode='valid', stride=(1, 1), filter_flip=True):
#     filterRow, filterCol = filter.shape
#     rowStride, colStride = stride
#     if border_mode == 'half':
#         X = pad2d(X, (filterRow // 2, filterRow // 2, filterCol // 2, filterCol // 2))
#     elif border_mode == 'full':
#         X = pad2d(X, (filterRow - 1, filterRow - 1, filterCol - 1, filterCol - 1))
#     nSample, nMap, mapRow, mapCol = X.shape
#     outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
#     X = X[index].reshape(nSample, nMap, outRow, outCol, filterRow, filterCol)
#     if filter_flip:
#         filter = filter[::-1, ::-1]
#     out = np.tensordot(X, filter, axes=[[-2, -1], [-2, -1]])
#     return out
#
#
# def pad2d(X, padding=(0, 0, 0, 0)):
#     inputShape = X.shape
#     outputShape = (inputShape[0],
#                    inputShape[1],
#                    inputShape[2] + padding[0] + padding[1],
#                    inputShape[3] + padding[2] + padding[3])
#     output = np.zeros(outputShape)
#     indices = (slice(None),
#                slice(None),
#                slice(padding[0], inputShape[2] + padding[0]),
#                slice(padding[2], inputShape[3] + padding[2]))
#     output[indices] = X
#     return output


# def genIndex1(XShape, filterShape, stride=(1, 1)):
#     nSample, nMap, mapRow, mapCol = XShape
#     filterRow, filterCol = filterShape
#     rowStride, colStride = stride
#     outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
#     block1 = np.arange(filterCol, dtype='int')
#     block2 = []
#     for i in range(filterRow):
#         block2.append(block1 + i * mapCol)
#     block2 = np.hstack(block2)
#     block3 = []
#     for i in range(outCol):
#         block3.append(block2 + i * colStride)
#     block3 = np.hstack(block3)
#     block4 = []
#     for i in range(outRow):
#         block4.append(block3 + i * mapCol * rowStride)
#     block4 = np.hstack(block4)
#     out = []
#     for i in range(nSample * nMap):
#         out.append(block4 + i * mapRow * mapCol)
#     return np.hstack(out).astype('int')
#
#
# # def genIndex2(XShape, filterShape, stride=(1, 1)):
# #     nSample, nMap, mapRow, mapCol = XShape
# #     filterRow, filterCol = filterShape
# #     rowStride, colStride = stride
# #     outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
# #     block1 = np.arange(filterCol, dtype='int')
# #     block2=np.tile(block1,filterRow)
# #     arange1=np.arange(filterRow,dtype='int')*mapCol
# #     added1=np.repeat(arange1,filterCol,axis=0)
# #     block2+=added1
# #     block3=np.tile(block2,outCol)
# #     arange2=np.arange(outCol,dtype='int')*colStride
# #     added2=np.repeat(arange2,filterRow*filterCol,axis=0)
# #     block3+=added2
# #     block4=np.tile(block3,outRow)
# #     arange3=np.arange(outRow)*mapCol*rowStride
# #     added3=np.repeat(arange3,filterRow*filterCol*outCol)
# #     block4+=added3
# #     out=np.tile(block4,nSample*nMap)
# #     arange4=np.arange(nSample*nMap,dtype='int')*mapRow*mapCol
# #     added4=np.repeat(arange4,outRow*outCol*filterRow*filterCol,axis=0)
# #     return out+added4
#
#
# def convdot2d1(X, filter, index, border_mode='valid', stride=(1, 1), filter_flip=True):
#     filterRow, filterCol = filter.shape
#     rowStride, colStride = stride
#     if border_mode == 'half':
#         X = pad2d1(X, (filterRow // 2, filterRow // 2, filterCol // 2, filterCol // 2))
#     elif border_mode == 'full':
#         X = pad2d1(X, (filterRow - 1, filterRow - 1, filterCol - 1, filterCol - 1))
#     nSample, nMap, mapRow, mapCol = X.shape
#     outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
#     X = X.reshape(-1)
#     X = X[index].reshape(nSample, nMap, outRow, outCol, filterRow, filterCol)
#     if filter_flip:
#         filter = filter[::-1, ::-1]
#     out = np.tensordot(X, filter, axes=[[-2, -1], [-2, -1]])
#     return out
#
#
# def pad2d1(X, padding=(0, 0, 0, 0)):
#     inputShape = X.shape
#     outputShape = (inputShape[0],
#                    inputShape[1],
#                    inputShape[2] + padding[0] + padding[1],
#                    inputShape[3] + padding[2] + padding[3])
#     output = np.zeros(outputShape)
#     indices = (slice(None),
#                slice(None),
#                slice(padding[0], inputShape[2] + padding[0]),
#                slice(padding[2], inputShape[3] + padding[2]))
#     output[indices] = X
#     return output
#
#
# a = np.arange(2800000).reshape((40, 20, 70, 50))
#
# f = np.arange(9).reshape((3, 3))

# tic = time.time()
# index = genIndex(a.shape, f.shape)
# print time.time() - tic
# for _ in range(10):
#     d1 = convdot2d(a, f, index)
# print time.time() - tic

# tic = time.time()
# index = genIndex1(a.shape, f.shape)
# print time.time() - tic
# for _ in range(10):
#     d2 = convdot2d1(a, f, index)
# print time.time() - tic
#
# from scipy.signal import convolve2d
# tic = time.time()
# d3 = np.zeros((40, 20, 68, 48))
# for _ in range(10):
#     for i in range(40):
#         for j in range(20):
#             d3[i][j] = convolve2d(a[i, j], f, mode='valid')
# print time.time() - tic
# print np.allclose(d2, d3)

def genIndex1(XShape, filterShape, border_mode='valid', stride=(1, 1)):
    nSample, nMap, mapRow, mapCol = XShape
    _, _, filterRow, filterCol = filterShape
    rowStride, colStride = stride
    if border_mode == 'half':
        mapRow += 2 * filterRow // 2
        mapCol += 2 * filterCol // 2
    elif border_mode == 'full':
        mapRow += 2 * (filterRow - 1)
        mapCol += 2 * (filterCol - 1)
    outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
    block1 = np.arange(filterCol, dtype='int')
    block2 = []
    for i in xrange(filterRow):
        block2.append(block1 + i * mapCol)
    block2 = np.hstack(block2)
    block3 = []
    for i in xrange(outCol):
        block3.append(block2 + i * colStride)
    block3 = np.hstack(block3)
    block4 = []
    for i in xrange(outRow):
        block4.append(block3 + i * mapCol * rowStride)
    block4 = np.hstack(block4)
    out = []
    for i in xrange(nSample * nMap):
        out.append(block4 + i * mapRow * mapCol)
    return np.hstack(out).astype('int')


def genIndex2(XShape, filterShape, border_mode='valid', stride=(1, 1)):
    nSample, nMap, mapRow, mapCol = XShape
    _, _, filterRow, filterCol = filterShape
    rowStride, colStride = stride
    if border_mode == 'half':
        mapRow += 2 * filterRow // 2
        mapCol += 2 * filterCol // 2
    elif border_mode == 'full':
        mapRow += 2 * (filterRow - 1)
        mapCol += 2 * (filterCol - 1)
    outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
    block1 = np.arange(filterCol, dtype='int')
    block2 = np.tile(block1, filterRow)
    arange1 = np.arange(filterRow, dtype='int') * mapCol
    added1 = np.repeat(arange1, filterCol, axis=0)
    block2 += added1
    block3 = np.tile(block2, outCol)
    arange2 = np.arange(outCol, dtype='int') * colStride
    added2 = np.repeat(arange2, filterRow * filterCol, axis=0)
    block3 += added2
    block4 = np.tile(block3, outRow)
    arange3 = np.arange(outRow) * mapCol * rowStride
    added3 = np.repeat(arange3, filterRow * filterCol * outCol)
    block4 += added3
    out = np.tile(block4, nSample * nMap)
    arange4 = np.arange(nSample * nMap, dtype='int') * mapRow * mapCol
    added4 = np.repeat(arange4, outRow * outCol * filterRow * filterCol, axis=0)
    return out + added4


def convdot2d(X, f, index, border_mode='valid', stride=(1, 1), filter_flip=True):
    nSample, nMap, mapRow, mapCol = X.shape
    _, _, filterRow, filterCol = f.shape
    rowStride, colStride = stride
    if border_mode == 'half':
        X = pad2d(X, (filterRow // 2, filterRow // 2, filterCol // 2, filterCol // 2))
        mapRow += 2 * (filterRow // 2)
        mapCol += 2 * (filterCol // 2)
    elif border_mode == 'full':
        X = pad2d(X, (filterRow - 1, filterRow - 1, filterCol - 1, filterCol - 1))
        mapRow += 2 * (filterRow - 1)
        mapCol += 2 * (filterCol - 1)
    outRow, outCol = (mapRow - filterRow) // rowStride + 1, (mapCol - filterCol) // colStride + 1
    X = T.flatten(X, outdim=1)
    X = X[index].reshape((nSample, nMap, outRow, outCol, filterRow, filterCol))
    if filter_flip:
        f = f[:, :, ::-1, ::-1]
    out = T.tensordot(X, f, axes=[[1, X.ndim - 2, X.ndim - 1], [1, f.ndim - 2, f.ndim - 1]])
    return out.dimshuffle(0, 3, 1, 2)


def pad2d(X, padding=(0, 0, 0, 0)):
    inputShape = X.shape
    outputShape = (inputShape[0],
                   inputShape[1],
                   inputShape[2] + padding[0] + padding[1],
                   inputShape[3] + padding[2] + padding[3])
    output = T.zeros(outputShape)
    indices = (slice(None),
               slice(None),
               slice(padding[0], inputShape[2] + padding[0]),
               slice(padding[2], inputShape[3] + padding[2]))
    return T.set_subtensor(output[indices], X)


a = np.arange(250880).reshape((10, 32, 28, 28))

f = np.arange(18432).reshape((64, 32, 3, 3))

x = T.tensor4()
y = T.tensor4()
index = genIndex1(a.shape, f.shape, stride=(2, 2))
z1 = convdot2d(x, y, index, border_mode='valid', stride=(2, 2))
f1 = function([x, y], z1, allow_input_downcast=True)
z2 = T.nnet.conv2d(x, y, subsample=(2, 2), border_mode='valid')
f2 = function([x, y], z2, allow_input_downcast=True)

tic = time.time()
for i in range(50):
    d1 = f1(a, f)
print time.time() - tic
tic = time.time()
for i in range(50):
    d2 = f2(a, f)
print time.time() - tic
print np.allclose(d1, d2)
# tic = time.time()
# for i in range(100):
#     index1 = genIndex1(a.shape, f.shape, stride=(1, 1))
# print time.time() - tic
# tic = time.time()
# for i in range(100):
#     index2 = genIndex2(a.shape, f.shape, stride=(1, 1))
# print time.time() - tic
# print np.allclose(index1, index2)
