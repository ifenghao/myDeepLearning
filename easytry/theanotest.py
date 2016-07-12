# coding:utf-8
__author__ = 'zfh'

import numpy as np
import theano
import theano.tensor as T
from theano import function

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
from compiler.ast import flatten
x=[[1,[2],3],[4,5,6]]
print flatten(x)
