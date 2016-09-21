# coding:utf-8
__author__ = 'zfh'

import numpy as np
import theano
import theano.tensor as T
from theano import function
import utils
import os, cPickle
import pylab
import time

# x=np.ones((10,1,3,3))
# y=np.ones((10,1,1,1))*2
#
# b=theano.shared(y,broadcastable=(False,True,True,True))
# xx=T.ftensor4()
# z=xx+b
# ff=function([xx],z,allow_input_downcast=True)
#
# print ff(x)

# yvalue=np.array([[1,2,3],[4,5,6]])
# X=T.ftensor3()
# y=theano.shared(yvalue)
# y=y.dimshuffle(0,1,'x')
# z=X+y
# # z=z.max(axis=1,keepdims=True)
# # z=z.dimshuffle(0,'x')
# f=function([X],z,allow_input_downcast=True)
# Xvalue=np.ones((2,3,4))
# print f(Xvalue)

# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# srng = RandomStreams()
# X=T.scalar()
# b=srng.binomial((10,10), p=0.9, dtype=theano.config.floatX)
# z=X*b
# f=function([X],z,allow_input_downcast=True)
# print f(1)
# print f(1)
# print f(1)

# from load import mnist
# import utils
# trX, teX, trY, teY = mnist(onehot=True)
# trX = trX.reshape(-1, 1, 28, 28)
# teX = teX.reshape(-1, 1, 28, 28)
#
# X = T.tensor4('X')
# prams = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
# wconv1 = utils.weightInit((32, 1, 5, 5), 'wconv1')
# bconv1 = utils.biasInit((32,), 'bconv1')
# prams.append([wconv1, bconv1])
# wconv2 = utils.weightInit((64, 32, 3, 3), 'wconv2')
# bconv2 = utils.biasInit((64,), 'bconv2')
# prams.append([wconv2, bconv2])
# wconv3 = utils.weightInit((128, 64, 3, 3), 'wconv3')
# bconv3 = utils.biasInit((128,), 'bconv3')
# prams.append([wconv3, bconv3])
# inputX=trX[:2].reshape(-1, 1, 28, 28)
# featureMaps= utils.listFeatureMap(inputX,prams)
# utils.showFeatureMap(featureMaps)

# import pylab
# pylab.subplot(1,3,1)
# pylab.imshow(np.random.rand(10,10)*255)
# pylab.subplot(1,3,2)
# pylab.imshow(np.random.rand(30,30)*255)
# pylab.subplot(1,3,3)
# pylab.imshow(np.random.rand(80,80)*255)
# pylab.gray()
# pylab.show()

# a1=np.ones((2,3,3))
# a2=np.ones((4,3,3))
# print np.vstack([a1,a2]).shape
# print np.split(a2,4,axis=0)
# print utils.squareStack(a2)

# from load import cifar
# import colorsys
# trX, teX, trY, teY=cifar()
# x=trX[0]
# x=np.stack((x[0],x[1],x[2]),axis=2)
# y=colorsys.rgb_to_hsv(x[0],x[1],x[2])
# pylab.imshow(x)
# pylab.show()
# fd = open(os.path.join('/home/zfh/dataset/cifar-10-batches-py/', 'data_batch_1'))
# dict = cPickle.load(fd)
# fd.close()
# image= dict['data'][:3]
# label=dict['labels']
# image=image.reshape(-1,3,32,32)
# image=np.stack((image[2][0],image[2][1],image[2][2]),axis=2)
# pylab.imshow(image)
# pylab.show()

# from sklearn.cross_validation import KFold
# X, y = np.arange(40).reshape((5,2, 2,2)), np.arange(10).reshape((5,2))
# kf=KFold(5,n_folds=2)
# for i,j in kf:
#     print X[i]
#     print y[i]
#     print X[j]
#     print y[j]

# def f():
#     a=[]
#     b=[]
#     while True:
#         new=(yield a,b)
#         if new is not None:
#             a.append(new[0])
#             b.append(new[1])
#
# a=f()
# print a.next()
# for i in range(10):
#     print a.send((i,i+10))


# x=np.arange(50)
# y=np.arange(50,100)
# gen=utils.earlyStopGen()
# gen.next()
# for i,j in zip(x,y):
#     print gen.send((i,j))

# x=np.ones((1,1,10,10))*2
# y=np.ones((1,1,3,3))
#
# X=T.tensor4()
# Y=T.tensor4()
# z=T.nnet.conv2d(X,Y,border_mode='half')
# ff=function([X,Y],z)
#
# print ff(x,y)

# x=np.arange(24).reshape((2,3,2,2))
# avg=np.mean(x,axis=0,dtype=float,keepdims=False)
# std=np.std(x,axis=0)
# print avg,std

# from theano.tensor.signal.pool import pool_2d
# x=np.ones((1,1,10,10))*2
#
# X=T.tensor4()
# z=pool_2d(X,(2,2),st=(1,1),ignore_border=True,padding=(1,1))
# ff=function([X],z)
#
# print ff(x)

# x=np.random.uniform(1,100,100)
# y=np.random.uniform(0,100,100)
# z=np.random.uniform(0,10,100)
# utils.scatter(x,y,z)

# import re
# pattern1=re.compile('(\(.*?\))')
# pattern2=re.compile('(\[.*?\])')
#
# lrList=[]
# cList=[]
# error=[]
# with open('/home/zfh/new') as file:
#     lines=file.read()
#     match1=pattern1.findall(lines)
#     match2=pattern2.findall(lines)
#     for m1,m2 in zip(match1,match2):
#         split=m1[1:-1].split(', ')
#         lrList.append(float(split[0]))
#         cList.append(float(split[1]))
#         error.append(float(m2[1:-1]))

# error=np.array(error)
# error=error[np.where(error<0.25)]
# lrList=np.array(lrList)
# lrList=lrList[np.where(error<0.25)]
# cList=np.array(cList)
# cList=cList[np.where(error<0.25)]
# utils.scatter(lrList,cList,error)

# x=np.arange(10).reshape((5,2))
# y=np.arange(24).reshape((4,2,3))
#
# X=T.matrix()
# Y=T.tensor3()
# z=T.dot(X,Y)
# ff=function([X,Y],[z,T.max(z,axis=1)])
#
# print x
# print y
# print ff(x,y)

# x=np.arange(60).reshape((5,4,3))
# b=np.arange(3)
#
# X=T.tensor3()
# B=theano.shared(b)
# z=X+B
# ff=function([X],z)
#
# print ff(x)

# x=np.arange(10).reshape((5,2))
# y=np.arange(24).reshape((4,2,3))
# print np.max(np.stack((y,y)),axis=0).shape

# x = T.matrix()
# splits = np.array([2, 2, 2])
# xsplit = T.split(x, splits, n_splits=3, axis=1)
# xmax = map(lambda x: T.max(x, axis=1), xsplit)
# xstack = T.stack(xmax, axis=1)
# f = function([x], xstack)
# print f(np.arange(24).reshape((4, 6)))

# x = T.matrix()
# xx=T.reshape(T.transpose(x),(3,2,4))
#
# xmax = T.max(xx, axis=1)
# xt = T.transpose(xmax)
# f = function([x], xt)
# a=np.arange(24).reshape((4, 6))
# print a,f(a)

# x=np.arange(120).reshape((2,3,4,5))
# y=np.arange(36).reshape((3,4,3))
# print x,y
# print np.tensordot(x,y,[[1,2],[0,1]]).shape

# x = T.tensor4()
# y = T.tensor4()
# xstack = T.concatenate((x,y),axis=0)
# f = function([x,y], xstack)
# a=np.arange(36).reshape((1,2,3, 6))
# print f(a,a).shape

# a=np.arange(120).reshape(2,3,4,5)
# b=np.arange(120,240).reshape(2,3,4,5)
# print a,b,np.sum(a,axis=0)

# x = T.tensor4()
# result=[]
# span=T.arange(x.shape[0])
# splits=T.ones_like(span,dtype='int')
# for i in T.split(span,splits,6):
#     result.append(x[i])
# f = function([x], result)
# a=np.arange(36).reshape((6,1,3, 2))
# print f(a)

# coefficients = T.vector("coefficients")
# x = T.scalar("x")
# indices=T.arange(coefficients.shape[0])
# components, updates = theano.scan(fn=lambda index, power,coefficients, x: coefficients[index] * (x ** power),
#                                   outputs_info=None,
#                                   sequences=[indices, T.arange(1000)],
#                                   non_sequences=[coefficients,x])
# print components
# polynomial = components
# calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)
# print calculate_polynomial([1,0,2],3)

# x = T.tensor4()
# index0=T.arange(x.shape[0])
# index0=T.repeat(index0,x.shape[1],axis=0)
# index1=T.arange(x.shape[1])
# index1=T.tile(index1,x.shape[0])
# result, updates = theano.scan(fn=lambda index0,index1, x:x[index0,index1],
#                                   outputs_info=None,
#                                   sequences=[index0,index1],
#                                   non_sequences=[x])
# f = theano.function(inputs=[x], outputs=result)
# print f(np.arange(240).reshape(10,4,2,3)).shape# scan返回按照第一维度拼接

# x = T.tensor4()
# span=T.arange(x.shape[0])
# xindex = x[2]
# f = function([x], xindex)
# a=np.arange(36).reshape((6,1,2,3))
# print f(a).shape

# a=np.arange(120).reshape((2,3,4,5))
# print np.transpose(a,axes=(1,0,2,3))
# a=a.reshape(2,3,1,4,5)
# print np.concatenate(list(a),axis=1)

# import h5py
# f=h5py.File('/home/zfh/downloadcode/googlenet/googlenet_weights.h5','r')
# data=f.get('conv1/7x7_s2/conv1')
# w,b= data.values()
# print np.array(w).shape,np.array(b).shape

# from scipy.misc import imread, imresize, imshow
# img = imread("/home/zfh/persian-cat.jpg", mode='RGB')
# img = imresize(img, (224, 224))
# img = np.array(img, dtype=np.float32)
# img = img[:, :, ::-1]
# img = img.transpose((2, 0, 1))
# avg=np.mean(img,axis=(1,2),keepdims=True)
# img-=avg
# imshow(img)

from keras.layers import Input, Dense, Convolution2D, BatchNormalization
from lasagne import layers
layers.DenseLayer
layers.Conv2DLayer
