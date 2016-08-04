# coding:utf-8
__author__ = 'zfh'

import numpy as np
import theano
import theano.tensor as T
from theano import function
import utils
import os,cPickle
import pylab

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