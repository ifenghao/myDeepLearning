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
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
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
    lconv1 = T.nnet.relu(conv2d(X, params[0][0], border_mode='full') +
                         params[0][1].dimshuffle('x', 0, 'x', 'x'))
    lds1 = max_pool_2d(lconv1, (2, 2))
    lds1 = utils.dropout(lds1, pDropConv)

    lconv2 = T.nnet.relu(conv2d(lds1, params[1][0]) +
                         params[1][1].dimshuffle('x', 0, 'x', 'x'))
    lds2 = max_pool_2d(lconv2, (2, 2))
    lds2 = utils.dropout(lds2, pDropConv)

    lconv3 = T.nnet.relu(conv2d(lds2, params[2][0]) +
                         params[2][1].dimshuffle('x', 0, 'x', 'x'))
    lds3 = max_pool_2d(lconv3, (2, 2))
    lds3 = utils.dropout(lds3, pDropConv)

    lflat = T.flatten(lds3, outdim=2)
    lfull = T.nnet.relu(T.dot(lflat, params[3][0]) + params[3][1])
    lfull = utils.dropout(lfull, pDropHidden)
    return softmax(T.dot(lfull, params[4][0]) + params[4][1])  # 如果使用nnet中的softmax训练出错


# 卷积网络，输入一组超参数，返回该网络的训练、验证、预测函数
class CConvNet(object):
    def __init__(self, fin, f1, f2, f3, hiddens, outputs,
                 lr=0.001, C=0.001, pDropConv=0.2, pDropHidden=0.5):
        # 待优化的权重和偏置shared变量
        self.params = []  # 所有需要优化的参数放入列表中，分别是连接权重和偏置
        # 卷积层，w=（本层特征图个数，上层特征图个数，卷积核行数，卷积核列数），b=（本层特征图个数）
        # conv: (32+3-1 , 32+3-1) = (34, 34)
        # pool: (34/2, 34/2) = (17, 17)
        wconv1 = utils.weightInit((f1, fin, 3, 3), 'wconv1')
        bconv1 = utils.biasInit((f1,), 'bconv1')
        self.params.append([wconv1, bconv1])
        # conv: (17-3+1 , 17-3+1) = (15, 15)
        # pool: (15/2, 15/2) = (8, 8)
        wconv2 = utils.weightInit((f2, f1, 3, 3), 'wconv2')
        bconv2 = utils.biasInit((f2,), 'bconv2')
        self.params.append([wconv2, bconv2])
        # conv: (8-3+1 , 8-3+1) = (6, 6)
        # pool: (6/2, 6/2) = (3, 3)
        wconv3 = utils.weightInit((f3, f2, 3, 3), 'wconv3')
        bconv3 = utils.biasInit((f3,), 'bconv3')
        self.params.append([wconv3, bconv3])
        # 全连接层，需要计算卷积最后一层的神经元个数作为MLP的输入
        wfull = utils.weightInit((f3 * 3 * 3, hiddens), 'wfull')
        bfull = utils.biasInit((hiddens,), 'bfull')
        self.params.append([wfull, bfull])
        wout = utils.weightInit((hiddens, outputs), 'wout')
        bout = utils.biasInit((outputs,), 'bout')
        self.params.append([wout, bout])

        # 定义 Theano 符号变量，并构建 Theano 表达式
        X = T.tensor4('X')
        Y = T.matrix('Y')
        YDropProb = model(X, self.params, pDropConv, pDropHidden)
        YFullProb = model(X, self.params, 0., 0.)
        YPred = T.argmax(YFullProb, axis=1)
        # 训练集代价函数
        trCrossEntropy = T.nnet.categorical_crossentropy(YDropProb, Y)
        trCost = T.mean(trCrossEntropy) + C * utils.reg(flatten(self.params))
        updates = utils.rmsprop(trCost, flatten(self.params), lr=lr)
        # 测试验证集代价函数
        vateCrossEntropy = T.nnet.categorical_crossentropy(YFullProb, Y)
        vateCost = T.mean(vateCrossEntropy) + C * utils.reg(flatten(self.params))

        # 编译函数
        # 训练函数，输入训练集，输出训练损失
        self.train = function(
            inputs=[In(X, borrow=True, allow_downcast=True),
                    In(Y, borrow=True, allow_downcast=True)],
            outputs=Out(trCost, borrow=True),  # 减少返回参数节省时间
            updates=updates,
            allow_input_downcast=True
        )
        # 验证函数，输入验证集，输出验证损失，不进行更新
        self.valtest = function(
            inputs=[In(X, borrow=True, allow_downcast=True),
                    In(Y, borrow=True, allow_downcast=True)],
            outputs=Out(vateCost, borrow=True),  # 减少返回参数节省时间
            allow_input_downcast=True
        )
        # 测试误差函数，输入数据集，输出错分类的误差
        self.error = function(
            inputs=[In(X, borrow=True, allow_downcast=True),
                    In(Y, borrow=True, allow_downcast=True)],
            outputs=Out(utils.errors(YFullProb, Y), borrow=True),  # 减少返回参数节省时间
            allow_input_downcast=True
        )
        # 预测函数，只输入X，输出预测结果
        self.predict = function(
            inputs=[In(X, borrow=True, allow_downcast=True)],
            outputs=Out(YPred, borrow=True),
            allow_input_downcast=True
        )

    # 训练卷积网络，最终返回在测试集上的误差
    def traincn(self, trX, teX, trY, teY, batchSize=200, maxIter=100,
                period=4, threshold=10, tol=2, verbose=True):
        earlyStop = utils.earlyStopGen(period, threshold, tol)
        earlyStop.next()  # 初始化生成器
        for epoch in range(maxIter):  # every epoch
            trCost = utils.miniBatchTrain(trX, trY, self.train, batchSize, verbose)
            teCost = self.valtest(teX, teY)
            if earlyStop.send((trCost, teCost)): break
            if verbose:
                print 'trError', self.error(trX, trY), \
                    'teError', self.error(teX, teY)

    # 交叉验证得到一组平均验证误差，使用早停止
    def cv(self, X, Y, folds=3, batchSize=200, maxIter=100,
           period=4, threshold=10, tol=2, verbose=True):
        kf = KFold(X.shape[0], n_folds=folds, random_state=42)  # 训练集分为3折交叉验证集
        vaCostList = []
        for trIndex, vaIndex in kf:
            trX, vaX, trY, vaY = X[trIndex], X[vaIndex], Y[trIndex], Y[vaIndex]
            earlyStop = utils.earlyStopGen(period, threshold, tol)
            earlyStop.next()  # 初始化生成器
            vaCostOpt = np.inf
            for epoch in range(maxIter):  # every epoch
                trCost = utils.miniBatchTrain(trX, trY, self.train, batchSize, verbose)
                vaCost = self.valtest(vaX, vaY)
                if vaCost < vaCostOpt: vaCostOpt = vaCost
                if earlyStop.send((trCost, vaCost)): break
            vaCostList.append(copy(vaCostOpt))
            if verbose: print '*' * 5, 'one validation done', '*' * 5
            self.resetPrams()
        return np.mean(vaCostList)

    # 重置优化参数，以重新训练模型
    def resetPrams(self):
        for p in self.params:
            utils.resetWeight(p[0])
            utils.resetBias(p[1])


def main():
    # 数据集，数据格式为4D矩阵（样本数，特征图个数，图像行数，图像列数）
    trX, teX, trY, teY = cifar(onehot=True)
    f1, f2, f3, hiddens = 32, 64, 128, 625
    params = utils.randomSearch(nIter=30)
    cvCostList = []
    for p in params:
        lr, C, pDropConv, pDropHidden = p
        print '*' * 10, 'parameters', p, '*' * 10
        convNet = CConvNet(fin=3, f1=f1, f2=f2, f3=f3, hiddens=hiddens, outputs=10,
                           lr=lr, C=C, pDropConv=pDropConv, pDropHidden=pDropHidden)
        cvCost = convNet.cv(trX, trY, folds=3, batchSize=200, maxIter=100,
                            period=3, threshold=10, tol=3, verbose=True)
        cvCostList.append(copy(cvCost))
    print '*' * 20, 'retraining', '*' * 20
    optIndex = np.argmin(cvCostList, axis=0)
    lr, C, pDropConv, pDropHidden = params[optIndex]
    convNet = CConvNet(fin=3, f1=f1, f2=f2, f3=f3, hiddens=hiddens, outputs=10,
                       lr=lr, C=C, pDropConv=pDropConv, pDropHidden=pDropHidden)
    convNet.traincn(trX, teX, trY, teY, batchSize=200, maxIter=100, verbose=True)


if __name__ == '__main__':
    main()
