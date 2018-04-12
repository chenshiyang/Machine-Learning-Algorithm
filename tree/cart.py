import numpy as np
from sklearn.datasets import load_boston

class TreeNode(object):
    def __init__(self, feature, val, left, right):
        self.featureToSplitOn = feature
        self.valOfSplit = val
        self.left = left
        self.right = right

class CART(object):
    def binSplitDataSet(self, dataset, feature, val):
        data1 = dataset[np.nonzero(dataset[:, feature] > val)[0], :][0]
        data2 = dataset[np.nonzero(dataset[:, feature] <= val)[0], :][0]
        return data1, data2

    def createTree(self, dataset, leafType = 'regLeaf', errType = 'regErr', ops=(1, 4)):
        feat, val = self.findBestSplit(dataset, leafType, errType, ops)
        if feat is None:
            return val
        leftData, rightData = self.binSplitDataSet(dataset, feat, val)
        leftTree = self.createTree(leftData, leafType, errType, ops)
        rightTree = self.createTree(rightData, leafType, errType, ops)
        treeNode = TreeNode(feat, val, leftTree, rightTree)
        return treeNode


    def findBestSplit(self, dataset, leafType, errType, ops):
        pass


if __name__ == '__main__':


