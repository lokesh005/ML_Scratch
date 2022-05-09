import numpy as np
from sklearn import datasets


class LinearRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __loss(self, h, y):
        return ((y-h)**2).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            h = np.dot(X, self.theta)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            h = np.dot(X, self.theta)
            loss = self.__loss(h, y)

            if (self.verbose == True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    def predict(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return np.dot(X, self.theta)


if __name__ == '__main__':
    diab = datasets.load_diabetes()

    X = diab.data[:, :2]
    y = diab.target

    model = LinearRegression(lr=0.1, num_iter=300000, verbose=True)
    model.fit(X, y)

    preds = model.predict(X)
    print(((preds - y)**2).mean())

    print(f"Intercept: {model.theta}")

