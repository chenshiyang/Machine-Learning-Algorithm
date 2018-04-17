import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
    ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],
    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: {} is not in the vocabulary.".format(word))
    return returnVec

def trainNB(trainMatrix, trainCategory):
    '''

    :param trainMatrix: 样本文本向量矩阵
    :param trainCategory: 样本标签
    :return:
    '''
    numTrain = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrain)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0#拉普拉斯平滑
    p1Denom = 2.0
    for i in range(numTrain):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i] #向量加法
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    #为了防止概率值太小,精度溢出,采用对数形式
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(testItem, p0Vec, p1Vec, pClass1):
    p1 = np.sum(testItem * p1Vec) + np.log(pClass1)
    p0 = np.sum(testItem * p0Vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testNB():
    posts, labels = loadDataSet()
    vocabList = createVocabList(posts)
    trainMat = []
    for post in posts:
        trainMat.append(setOfWords2Vec(vocabList, post))
    p0Vec, p1Vec, pA = trainNB(trainMat, labels)
    testSentence = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabList, testSentence))
    print("{} classified as {}".format( testSentence, classifyNB(thisDoc, p0Vec, p1Vec, pA)))

    testSentence2 = ['love', 'stupid', 'garbage']
    thisDoc2 = np.array(setOfWords2Vec(vocabList, testSentence2))
    print("{} classified as {}".format(testSentence2, classifyNB(thisDoc2, p0Vec, p1Vec, pA)))

if __name__ == '__main__':
    testNB()