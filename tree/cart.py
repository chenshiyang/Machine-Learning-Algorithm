import numpy as np
from sklearn.datasets import load_boston

class TreeNode(object):
    def __init__(self, feature, val, left, right):
        self.featureToSplitOn = feature
        self.valOfSplit = val
        self.left = left
        self.right = right

def regLeaf(dataset):
    '''
    计算给定叶子中样本标签值的平均值，作为叶子的值
    :param dataset:
    :return:
    '''
    return np.mean(dataset[:,-1])

def regErr(dataset):
    '''
    计算给定数据集合样本标签值的Sum of Square Error
    :param dataset:
    :return:
    '''
    return np.var(dataset[:,-1] * np.shape(dataset)[0])

class CART(object):
    tree = None
    def binSplitDataSet(self, dataset, feature, val):
        data1 = dataset[np.nonzero(dataset[:, feature] > val)[0], :]
        data2 = dataset[np.nonzero(dataset[:, feature] <= val)[0], :]
        return data1, data2

    def createTree(self, dataset, leafType = regLeaf, errType = regErr, ops=(1, 4)):
        feat, val = self.findBestSplit(dataset, leafType, errType, ops)
        if feat is None:
            return val
        leftData, rightData = self.binSplitDataSet(dataset, feat, val)
        leftTree = self.createTree(leftData, leafType, errType, ops)
        rightTree = self.createTree(rightData, leafType, errType, ops)
        treeNode = TreeNode(feat, val, leftTree, rightTree)
        return treeNode


    def findBestSplit(self, dataset, leafType = regLeaf, errType = regErr, ops=(1, 4)):
        tolS = ops[0]
        tolN = ops[1]
        if len(set(dataset[:, -1].T.tolist()[0])) == 1:
            return None, leafType(dataset)
        m, n = np.shape(dataset)
        S = errType(dataset)
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(n - 1):
            for splitVal in set(dataset[:, featIndex].T.tolist()[0]):
                data0, data1 = self.binSplitDataSet(dataset, featIndex, splitVal)
                #当分裂后每个子节点的样本个数小于阈值，则不分裂
                if np.shape(data0)[0] < tolN or np.shape(data1)[0] < tolN:
                    continue
                newS = errType(data0) + errType(data1)
                if newS < bestS:
                    bestS = newS
                    bestIndex = featIndex
                    bestValue = splitVal
        # 当Sum of Square Error的降低低于阈值，则不分裂
        if S - bestS < tolS:
            return None, leafType(dataset)
        return bestIndex, bestValue

    def fit(self, dataset, leafType = regLeaf, errType = regErr, ops=(1, 4) ):
        self.tree = self.createTree(dataset, leafType, errType, ops)

    def predict(self, testData):
        y_pred = []
        for item in testData:
            node = self.tree
            while isinstance(node, TreeNode):
                if item[node.featureToSplitOn] > node.valOfSplit:
                    node = node.left
                else:
                    node = node.right
            y_pred.append(node)
        return y_pred


if __name__ == '__main__':
    # testMat = np.mat(np.eye(4))
    # cart = CART()
    # mat0, mat1 = cart.binSplitDataSet(testMat, 1, 0.5)
    # print(mat0)
    # print(mat1)

    #加载波士顿数据
    lb = load_boston()
    data = lb.data.reshape((np.shape(lb.data)[0], -1))
    target = lb.target.reshape((-1 ,1))
    dataset = np.mat(np.hstack((data, target)))
    cart = CART()
    cart.fit(dataset)

    testData = data[0:50,:]
    y_pred = cart.predict(testData)
    for i in range(len(y_pred)):
        print(target[i,-1], y_pred[i])