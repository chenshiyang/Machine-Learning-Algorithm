from math import log
import operator

def calcShannonEnt(dataSet):
    '''
    计算输入的数据集的香农熵
    :param dataSet:
    :return:
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for item in dataSet:
        label = item[-1]
        if label not in labelCounts:
            labelCounts[label] = 0
        labelCounts[label] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        props = labelCounts[key] / numEntries
        shannonEnt += -props * log(props, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def splitDataSet(dataset, axis, value):
    '''
    从给定数据集生成特征等于给定值的子集
    :param dataset:
    :param axis:
    :param value:
    :return:
    '''
    retDataSet = []
    for item in dataset:
        if item[axis] == value:
            retItem = item[:axis]
            retItem.extend(item[axis + 1:])
            retDataSet.append(retItem)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    从给定数据集中选出信息增益最大的特征,返回特征的索引
    :param dataSet:
    :return:
    '''
    numFeature = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeature):
        featureVals = [item[i] for item in dataSet]
        uniqueFeatureVals = set(featureVals)
        newEntropy = 0.0
        for featVal in uniqueFeatureVals:
            subData = splitDataSet(dataSet, i, featVal)
            newEntropy += 1.0 * len(subData) / len(dataSet) * calcShannonEnt(subData)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature

def majorityCnt(classList):
    '''
    多数投票,返回频率最高的类别
    :param classList:
    :return:
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCnt = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCnt[0][0]

def createTree(dataSet, labels):
    classList = [item[-1] for item in dataSet]
    #如果所有样本都属于同一类别,直接返回该类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果当前没有特征可以用来划分,直接采用多数投票
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    tree = {bestFeatureLabel: {}}
    featVals = [item[bestFeature] for item in dataSet]
    uniqueFeatVals = set(featVals)
    del(labels[bestFeature])
    for val in uniqueFeatVals:
        subLabels = labels[:]
        tree[bestFeatureLabel][val] = createTree(splitDataSet(dataSet, bestFeature, val), subLabels)
    return tree

def classify(tree, featureLabels, testVec):
    '''
    根据训练好的树tree,预测样本testVec的类别
    :param tree:
    :param featureLabels:
    :param testVec:
    :return:
    '''
    while(type(tree).__name__ == 'dict'):
        featureName = list(tree.keys())[0]
        featureVals = tree[featureName]
        featureIndex = featureLabels.index(featureName)
        for key in featureVals.keys():
            if key == testVec[featureIndex]:
                tree = featureVals[key]
    return tree

def storeTree(tree, fileName):
    '''
    将决策树以字节的方式持久化到本地文件中
    :param tree:
    :param fileName:
    :return:
    '''
    import pickle
    f = open(fileName, 'wb')
    pickle.dump(tree, f)
    f.close()

def getTree(fileName):
    '''
    以字节读取的方式从本地文件中恢复决策树
    :param fileName:
    :return:
    '''
    import pickle
    with open(fileName, 'rb') as f:
        tree = pickle.load(f)
        return tree

if __name__ == '__main__':
    pass