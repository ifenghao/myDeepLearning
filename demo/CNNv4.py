# coding:utf-8
__author__ = 'zfh'
'''
训练技巧：
1、使用线性修正单元（relu）作为激活函数
2、每一层加入dropout
3、参数更新方式采用rmsprop
4、使用mini-batch分批训练
5、使用borrow=True属性
6、基于v1的结构增加了验证集上的超参数选取
'''
from compiler.ast import flatten
import time
import theano.tensor as T
from theano import function, In, Out
from theano.tensor.nnet import conv2d, categorical_crossentropy, relu
from theano.tensor.signal.pool import pool_2d
from sklearn.cross_validation import KFold
from copy import copy
import numpy as np
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
def model(X, params, pDropConv, pDropHidden):
    layer = 0
    lconv = relu(conv2d(X, params[layer][0], border_mode='half') +
                 params[layer][1].dimshuffle('x', 0, 'x', 'x'))
    lconv = pool_2d(lconv, (2, 2), ignore_border=False)
    lconv = utils.dropout(lconv, pDropConv)
    # 32
    layer += 1
    lconv = relu(conv2d(lconv, params[layer][0], border_mode='half') +
                 params[layer][1].dimshuffle('x', 0, 'x', 'x'))
    lconv = utils.dropout(lconv, pDropConv)

    layer += 1
    lconv = relu(conv2d(lconv, params[layer][0], border_mode='half') +
                 params[layer][1].dimshuffle('x', 0, 'x', 'x'))
    lconv = pool_2d(lconv, (2, 2), ignore_border=False)
    lconv = utils.dropout(lconv, pDropConv)
    # 64
    layer += 1
    lconv = relu(conv2d(lconv, params[layer][0], border_mode='half') +
                 params[layer][1].dimshuffle('x', 0, 'x', 'x'))
    lconv = utils.dropout(lconv, pDropConv)

    layer += 1
    lconv = relu(conv2d(lconv, params[layer][0], border_mode='half') +
                 params[layer][1].dimshuffle('x', 0, 'x', 'x'))
    lconv = pool_2d(lconv, (2, 2), ignore_border=False)
    lconv = utils.dropout(lconv, pDropConv)
    # 128
    layer += 1
    lflat = T.flatten(lconv, outdim=2)
    lfull = relu(T.dot(lflat, params[layer][0]) + params[layer][1])
    lfull = utils.dropout(lfull, pDropHidden)

    layer += 1
    lfull = relu(T.dot(lfull, params[layer][0]) + params[layer][1])
    lfull = utils.dropout(lfull, pDropHidden)

    layer += 1
    return softmax(T.dot(lfull, params[layer][0]) + params[layer][1])  # 如果使用nnet中的softmax训练产生NAN


# 卷积网络，输入一组超参数，返回该网络的训练、验证、预测函数
class CConvNet(object):
    def __init__(self, fin, f1, f2, f3, f4, f5, h1, h2, outputs,
                 lr=0.001, C=0.001, pDropConv=0., pDropHidden=0.5):
        self.params = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
        # 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
        # conv: (32, 32) = (32, 32)
        # pool: (32/2, 32/2) = (16, 16)
        wconv1 = utils.weightInit2((f1, fin, 3, 3), 'wconv1')
        bconv1 = utils.biasInit((f1,), 'bconv1')
        self.params.append([wconv1, bconv1])
        # conv: (16, 16) = (16, 16)
        wconv2 = utils.weightInit2((f2, f1, 3, 3), 'wconv2')
        bconv2 = utils.biasInit((f2,), 'bconv2')
        self.params.append([wconv2, bconv2])
        # conv: (16, 16) = (16, 16)
        # pool: (16/2, 16/2) = (8, 8)
        wconv3 = utils.weightInit2((f3, f2, 3, 3), 'wconv3')
        bconv3 = utils.biasInit((f3,), 'bconv3')
        self.params.append([wconv3, bconv3])
        # conv: (8, 8) = (8, 8)
        wconv4 = utils.weightInit2((f4, f3, 3, 3), 'wconv4')
        bconv4 = utils.biasInit((f4,), 'bconv4')
        self.params.append([wconv4, bconv4])
        # conv: (8, 8) = (8, 8)
        # pool: (8/2, 8/2) = (4, 4)
        wconv5 = utils.weightInit2((f5, f4, 3, 3), 'wconv5')
        bconv5 = utils.biasInit((f5,), 'bconv5')
        self.params.append([wconv5, bconv5])
        # 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
        wfull1 = utils.weightInit2((f5 * 4 * 4, h1), 'wfull1')
        bfull1 = utils.biasInit((h1,), 'bfull1')
        self.params.append([wfull1, bfull1])
        wfull2 = utils.weightInit2((h1, h2), 'wfull2')
        bfull2 = utils.biasInit((h2,), 'bfull2')
        self.params.append([wfull2, bfull2])
        wout = utils.weightInit2((h2, outputs), 'wout')
        bout = utils.biasInit((outputs,), 'bout')
        self.params.append([wout, bout])

        # 定义 Theano 符号变量，并构建 Theano 表达式
        Xtr = T.tensor4('Xtr')
        Ytr = T.matrix('Ytr')
        YDropProb = model(Xtr, self.params, pDropConv, pDropHidden)
        # 训练集代价函数
        trCrossEntropy = categorical_crossentropy(YDropProb, Ytr)
        trCost = T.mean(trCrossEntropy) + C * utils.reg(flatten(self.params))
        updates = utils.rmsprop(trCost, flatten(self.params), lr=lr)

        Xvt = T.tensor4('Xvt')
        Yvt = T.matrix('Yvt')
        YFullProb = model(Xvt, self.params, 0., 0.)
        YPred = T.argmax(YFullProb, axis=1)
        # 测试验证集代价函数
        vateCrossEntropy = categorical_crossentropy(YFullProb, Yvt)
        vateCost = T.mean(vateCrossEntropy) + C * utils.reg(flatten(self.params))

        # 编译函数
        # 训练函数，输入训练集，输出训练损失和误差
        self.train = function(
            inputs=[In(Xtr, borrow=True, allow_downcast=True),
                    In(Ytr, borrow=True, allow_downcast=True)],
            outputs=[Out(trCost, borrow=True),
                     Out(utils.neqs(YDropProb, Ytr), borrow=True)],  # 减少返回参数节省时间
            updates=updates,
            allow_input_downcast=True
        )
        # 验证或测试函数，输入验证或测试集，输出损失和误差，不进行更新
        self.valtest = function(
            inputs=[In(Xvt, borrow=True, allow_downcast=True),
                    In(Yvt, borrow=True, allow_downcast=True)],
            outputs=[Out(vateCost, borrow=True),
                     Out(utils.neqs(YFullProb, Yvt), borrow=True)],  # 减少返回参数节省时间
            allow_input_downcast=True
        )
        # 预测函数，只输入X，输出预测结果
        self.predict = function(
            inputs=[In(Xvt, borrow=True, allow_downcast=True)],
            outputs=Out(YPred, borrow=True),
            allow_input_downcast=True
        )

    # 训练卷积网络，最终返回在测试集上的误差
    def traincn(self, trX, teX, trY, teY, batchSize=128, maxIter=100,
                start=30, period=4, threshold=10, tol=2, verbose=True):
        earlyStop = utils.earlyStopGen(start, period, threshold, tol)
        earlyStop.next()  # 初始化生成器
        for epoch in range(maxIter):  # every epoch
            startTime = time.time()
            trCost = utils.miniBatch(trX, trY, self.train, batchSize, verbose)
            teCost = utils.miniBatch(teX, teY, self.valtest, batchSize, verbose)
            if earlyStop.send((trCost, teCost)): break
            if verbose: print ' time: %10.5f' % (time.time() - startTime)

    # 交叉验证得到一组平均验证误差，使用早停止
    def cv(self, X, Y, folds=3, batchSize=128, maxIter=100,
           start=30, period=4, threshold=10, tol=2, verbose=True):
        kf = KFold(X.shape[0], n_folds=folds, random_state=42)  # 训练集分为3折交叉验证集
        vaCostList = []
        for trIndex, vaIndex in kf:
            trX, vaX, trY, vaY = X[trIndex], X[vaIndex], Y[trIndex], Y[vaIndex]
            earlyStop = utils.earlyStopGen(start, period, threshold, tol)
            earlyStop.next()  # 初始化生成器
            vaCostOpt = np.inf
            for epoch in range(maxIter):  # every epoch
                startTime = time.time()
                trCost = utils.miniBatch(trX, trY, self.train, batchSize, verbose)
                vaCost = utils.miniBatch(vaX, vaY, self.valtest, batchSize, verbose)
                if vaCost < vaCostOpt: vaCostOpt = vaCost
                if earlyStop.send((trCost, vaCost)): break
                if verbose: print ' time: %10.5f' % (time.time() - startTime)
            vaCostList.append(copy(vaCostOpt))
            if verbose: print '*' * 5, 'one validation done', '*' * 5
            self.resetPrams()
        return np.mean(vaCostList)

    # 重置优化参数，以重新训练模型
    def resetPrams(self):
        for p in self.params:
            utils.resetWeight2(p[0])
            utils.resetBias(p[1])


def main():
    # 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
    trX, teX, trY, teY = cifar(onehot=True)
    f1, f2, f3, f4, f5, h1, h2 = 32, 64, 64, 128, 128, 1024, 1024
    params = utils.randomSearch(nIter=10)
    cvCostList = []
    for param, num in zip(params, range(len(params))):
        lr, C = param
        print '*' * 10, num + 1, 'parameters', param, '*' * 10
        convNet = CConvNet(3, f1, f2, f3, f4, f5, h1, h2, 10, lr, C, 0, 0.5)
        cvCost = convNet.cv(trX, trY, folds=3, batchSize=128, maxIter=100,
                            start=30, period=4, threshold=10, tol=2, verbose=True)
        cvCostList.append(copy(cvCost))
    optIndex = np.argmin(cvCostList, axis=0)
    lr, C = params[optIndex]
    print 'retraining', params[optIndex]
    convNet = CConvNet(3, f1, f2, f3, f4, f5, h1, h2, 10, lr, C, 0, 0.5)
    convNet.traincn(trX, teX, trY, teY, batchSize=128, maxIter=100,
                    start=30, period=4, threshold=10, tol=2, verbose=True)


if __name__ == '__main__':
    main()
