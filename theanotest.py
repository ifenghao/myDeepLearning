# coding:utf-8
__author__ = 'zfh'

import numpy as np
from theano import *
import theano.tensor as T
from theano import function

x=T.dscalar('x')
y=T.dscalar('y')
z=x+y
f=function([x,y],z)