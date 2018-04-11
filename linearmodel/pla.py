# -*-encoding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def createLinearSeparableData(W, m=100, seed = 100):

    '''
    :param W: 目标函数f的法向量
    :param m: 需要产生的样本个数
    :return: X:np.array 生成的数据
    '''
    if W is None or len(W) == 0:
        return None
    np.random.seed(seed)
    n = len(W)
    w = np.array(W)
    D = np.zeros(shape=(m, n+1))
    for i in range(0, m):
        x = np.random.rand(1, n) * 20 - 10
        t = np.sum(w * x)
        if t >= 0:
            D[i] = np.append(x, 1)
        else:
            D[i] = np.append(x, -1)
    return D

def plotData(D):
    fig = plt.figure()
    sub1 = fig.add_subplot(1, 1, 1)
    sub1.set_title('Data plot')
    plt.xlabel('f1')
    plt.ylabel('f2')
    pos_idx = np.where(D[:, -1] == 1)
    neg_idx = np.where(D[:, -1] == -1)
    sub1.scatter(x = D[pos_idx, 0], y = D[pos_idx, 1], marker = 'o', color = 'y')
    sub1.scatter(x = D[neg_idx, 0], y = D[neg_idx, 1], marker = 'x', color = 'r')
    plt.legend(loc = 'upper right')
    plt.show()
class Perceptron(object):
    w = None

    def fit(self, Data, draw = False):
        '''
        :param Data: np.array
        :param draw:
        :return:
        '''
        m = Data.shape[0]
        n = Data.shape[1] - 1
        w = np.zeros((1, n), dtype = np.float64)
        separated = False
        i = 0
        while not separated and i < m:
            if Data[i][-1] * np.sum(w * Data[i][0:-1]) <= 0:
                separated = False
                w = w + Data[i][-1] * Data[i][0: -1]
            else:
                if i == m - 1:
                    separated = True
                i += 1
        self.w = w
        if draw:
            from matplotlib.lines import Line2D
            fig = plt.figure()
            sub1 = fig.add_subplot(111)
            sub1.set_title('Data plot')
            plt.xlabel('f1')
            plt.ylabel('f2')
            pos_idx = np.where(D[:, -1] == 1)
            neg_idx = np.where(D[:, -1] == -1)
            sub1.scatter(x=D[pos_idx, 0], y=D[pos_idx, 1], marker='o', color='y')
            sub1.scatter(x=D[neg_idx, 0], y=D[neg_idx, 1], marker='x', color='r')
            plt.legend(loc='upper right')
            #法向量w
            print(w)
            wx = w[0][0] / abs(w[0][0]) * 10
            wy = w[0][1] / abs(w[0][1]) * 10
            sub1.annotate("", xy = (wx, wy), xytext = (0, 0), size = 20, arrowprops = dict(arrowstyle='-|>'))
            ys = (-12 * (-w[0][0]) / w[0][1], 12 * (-w[0][0]) / w[0][1])
            #分隔线
            sub1.add_line(Line2D((-12, 12), ys, linewidth = 1, color = 'blue'))
            plt.show()

    def predict(self, testData):
        '''
        输入一个二维list,输出预测后的类别
        :param testData:
        :return:
        '''
        if self.w is None:
            print("Must fit before predict")
            return None
        result = []
        for x in testData:
            x = np.array(x)
            if np.sum(self.w * x) >= 0:
                result.append(1)
            else:
                result.append(-1)

        return result



if __name__ == '__main__':
    D = createLinearSeparableData([10, 10])
    # plotData(D)
    pla = Perceptron()
    pla.fit(D, draw = True)
    y = pla.predict([[5, 5]])
    print(y)
    print(pla.predict([[4, -2], [4, -6]]))

