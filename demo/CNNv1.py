# coding:utf-8
__author__ = 'zfh'
'''
训练技巧：
1、使用线性修正单元（relu）作为激活函数
2、每一层加入dropout
3、参数更新方式采用rmsprop
4、使用mini-batch分批训练
5、使用borrow=True属性
'''
from compiler.ast import flatten
import time
from copy import copy
import theano.tensor as T
from theano import function, In, Out
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import pylab
from load import cifar
import utils


# dimshuffle维度重排，将max得到的一维向量扩展成二维矩阵，第二维维度为1，也可以用[:,None]
def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


# 模型构建，返回给定样本判定为某类别的概率
# dimshuffle在偏置插入维度使之与相加矩阵相同（1，本层特征图个数，1，1），插入维度的broadcastable=True
# 每次调用dropout的模式都不同，即在每轮训练中网络结构都不同
# 本层的每个特征图和上层的所有特征图连接，可以不用去选择一些组合来部分连接
def model(X, prams, pDropConv, pDropHidden):
    lconv1 = T.nnet.relu(conv2d(X, prams[0][0], border_mode='full') +
                         prams[0][1].dimshuffle('x', 0, 'x', 'x'))
    lds1 = max_pool_2d(lconv1, (2, 2))
    lds1 = utils.dropout(lds1, pDropConv)

    lconv2 = T.nnet.relu(conv2d(lds1, prams[1][0]) +
                         prams[1][1].dimshuffle('x', 0, 'x', 'x'))
    lds2 = max_pool_2d(lconv2, (2, 2))
    lds2 = utils.dropout(lds2, pDropConv)

    lconv3 = T.nnet.relu(conv2d(lds2, prams[2][0]) +
                         prams[2][1].dimshuffle('x', 0, 'x', 'x'))
    lds3 = max_pool_2d(lconv3, (2, 2))
    lds3 = utils.dropout(lds3, pDropConv)

    lflat = T.flatten(lds3, outdim=2)
    lfull = T.nnet.relu(T.dot(lflat, prams[3][0]) + prams[3][1])
    lfull = utils.dropout(lfull, pDropHidden)
    return softmax(T.dot(lfull, prams[4][0]) + prams[4][1])  # 如果使用nnet中的softmax训练出错

# 常量
iterSteps = 100
learningRate = 0.001
C = 0.001
fin = 3
f1 = 32
f2 = 64
f3 = 128
hiddens = 625
outputs = 10
batchSize = 200

# 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
trX, teX, trY, teY = cifar(onehot=True)
trSize = trX.shape[0]  # 训练集样本数
teSize = teX.shape[0]  # 测试集样本数

# Theano 符号变量
X = T.tensor4('X')
Y = T.matrix('Y')
prams = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
# 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
# conv: (32+3-1 , 32+3-1) = (34, 34)
# pool: (34/2, 34/2) = (17, 17)
wconv1 = utils.weightInit((f1, fin, 3, 3), 'wconv1')
bconv1 = utils.biasInit((f1,), 'bconv1')
prams.append([wconv1, bconv1])
# conv: (17-3+1 , 17-3+1) = (15, 15)
# pool: (15/2, 15/2) = (8, 8)
wconv2 = utils.weightInit((f2, f1, 3, 3), 'wconv2')
bconv2 = utils.biasInit((f2,), 'bconv2')
prams.append([wconv2, bconv2])
# conv: (8-3+1 , 8-3+1) = (6, 6)
# pool: (6/2, 6/2) = (3, 3)
wconv3 = utils.weightInit((f3, f2, 3, 3), 'wconv3')
bconv3 = utils.biasInit((f3,), 'bconv3')
prams.append([wconv3, bconv3])
# 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
wfull = utils.weightInit2MLP((f3 * 3 * 3, hiddens), 'wfull')
bfull = utils.biasInit((hiddens,), 'bfull')
prams.append([wfull, bfull])
wout = utils.weightInit2MLP((hiddens, outputs), 'wout')
bout = utils.biasInit((outputs,), 'bout')
prams.append([wout, bout])

# 构建 Theano 表达式
YDropProb = model(X, prams, 0.2, 0.5)
YFullProb = model(X, prams, 0., 0.)
YPred = T.argmax(YFullProb, axis=1)
crossEntropy = T.nnet.categorical_crossentropy(YDropProb, Y)
cost = T.mean(crossEntropy) + C * utils.reg(flatten(prams))
updates = utils.rmsprop(cost, flatten(prams), lr=learningRate)

# 编译函数
# 训练函数，输入训练集，输出测试误差
train = function(
    inputs=[In(X, borrow=True, allow_downcast=True),
            In(Y, borrow=True, allow_downcast=True)],
    outputs=Out(utils.errors(YDropProb, Y), borrow=True),  # 减少返回参数节省时间
    updates=updates,
    allow_input_downcast=True
)
# 测试或验证函数，输入测试或验证集，输出测试或验证误差，不进行更新
test = function(
    inputs=[In(X, borrow=True, allow_downcast=True),
            In(Y, borrow=True, allow_downcast=True)],
    outputs=Out(utils.errors(YFullProb, Y), borrow=True),  # 减少返回参数节省时间
    allow_input_downcast=True
)
# 预测函数，只输入X，输出预测结果
predict = function(
    inputs=[In(X, borrow=True, allow_downcast=True)],
    outputs=Out(YPred, borrow=True),
    allow_input_downcast=True
)

# 训练迭代，一次迭代分为多batch训练
errorTrace = []
start = time.time()
for i in range(iterSteps):
    epochStart = time.time()
    for start, end in zip(range(0, trSize, batchSize), range(batchSize, trSize, batchSize)):
        trError = train(trX[start:end], trY[start:end])
        print 'trError:', trError, '\r',
    teError = test(teX, teY)
    errorTrace.append(copy(teError))
    print 'teError:', teError, 'time delay:', time.time() - epochStart
print 'total time:', time.time() - start

pylab.plot(errorTrace, 'b-')
pylab.show()

featureMaps = utils.listFeatureMap(trX[:5], prams)
utils.showFeatureMap(featureMaps)
