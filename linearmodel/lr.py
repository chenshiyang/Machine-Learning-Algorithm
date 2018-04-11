import numpy as np

class LogisticsRegression(object):
    '''
    实现二分类逻辑回归算法
    '''
    def __init__(self):
        self.W = None

    def fit(self, X, y, alpha = 0.001, num_iter = 5000):
        m, n = X.shape
        #init W
        self.W = np.random.randn(n, 1).reshape((-1, 1))
        #loss
        loss = []

        for i in range(num_iter):
            error, dW = self.compute_loss(X, y)
            self.W = self.W - alpha * dW

            loss.append(error)
            if i % 100 == 0:
                print('Round %d , error = %f' % (i, error))
        return loss

    def compute_loss(self, X, y):
        num_train = X.shape[0]
        h = self.h(X)
        loss = - 1 / num_train * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        dW = X.T.dot(h - y) / num_train
        return loss, dW

    def h(self, X):
        z = np.dot(X, self.W)
        return self.sigmoid(z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X_test, threshold = 0.5):
        h = self.h(X_test)
        y_pred = np.where(h >= threshold, 0, 1)
        return h, y_pred

    def __str__(self):
        return str(self.W)

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt
    iris = load_iris()
    X = iris.data[0:100,:]
    y = iris.target[0:100].reshape((-1, 1))
    one = np.ones((X.shape[0], 1))
    X_train = np.hstack((one, X))
    lr = LogisticsRegression()
    loss = lr.fit(X_train, y, num_iter = 10000)
    print(lr)
    plt.plot(loss)
    plt.xlabel('Iter num')
    plt.ylabel('loss')
    plt.show()

    h, y_pred = lr.predict(X_train, y)
    print(y)
    print('*' * 50)
    for i in range(h.shape[0]):
        print(h[i, 0]),
        print (y_pred[i, 0])
    print(np.sum(np.abs(y - y_pred)) / y.shape[0])

