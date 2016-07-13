# coding:utf-8
__author__ = 'zfh'

import numpy as np
import theano
import theano.tensor as T
from theano import function
import utils

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

from load import cifar
trX, teX, trY, teY=cifar()
print trX.shape, teX.shape, trY.shape, teY.shape
# import os,cPickle
# import pylab
# fd = open(os.path.join('/home/zfh/dataset/cifar-10-batches-py/', 'data_batch_1'))
# dict = cPickle.load(fd)
# fd.close()
# image= dict['data'][:3]
# label=dict['labels']
# image=image.reshape(-1,3,32,32)
# image=np.stack((image[2][0],image[2][1],image[2][2]),axis=2)
# pylab.imshow(image)
# pylab.show()
