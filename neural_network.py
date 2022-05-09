import numpy as np


def MSE(y, y_pred):
    t = 0.0
    for i in range(len(y)):
        t = t + (y_pred[i]-y[i])**2
    return t/(2*len(y))


class MLPNew:

    def __init__(self, input_feature_size, hidden_layer_size, outputs_layer_size, lr, epochs):

        self.input_feature_size = input_feature_size
        self.hidden_layer_size = hidden_layer_size
        self.outputs_layer_size = outputs_layer_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = 500
        self.error = []
        layers = [self.input_feature_size] + self.hidden_layer_size + [self.outputs_layer_size]

        # initiate random weights
        self.weights = []
        self.bias = []
        for i in range(1, len(layers)):
            w = np.random.rand(layers[i - 1], layers[i]) * np.sqrt(1. / layers[i])
            self.weights.append(w)

            b = np.random.rand(layers[i], 1)
            self.bias.append(b)

    def h(self, x):
        y = 1.0 / (1. + np.exp(-x))

        return y

    def hdf(self, x):
        return self.h(x) * (1 - self.h(x))

    def forward_propagate(self, X):
        self.zs = []
        self.activations = []

        for i in range(len(self.weights)):
            if i == 0:
                z_i = np.matmul((self.weights[i]).T, X) + self.bias[i]
            else:
                z_i = np.matmul((self.weights[i]).T, self.activations[i - 1]) + self.bias[i]

            activation = self.h(z_i)
            self.zs.append(z_i)
            self.activations.append(activation)

        # Since this is regression problem
        self.activations[-1] = self.zs[-1]
        return self.activations[-1]

    def back_propagate(self, X, Y):
        m = X.shape[0]
        self.dzs = [0 for i in range(len(self.weights))]
        self.dws = [0 for i in range(len(self.weights))]
        self.dbs = [0 for i in range(len(self.weights))]
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dz_i = (self.activations[i] - np.array([[Y]]))
            else:
                dz_i = np.matmul(self.weights[i + 1], self.dzs[i + 1]) * self.hdf(self.activations[i])

            self.dzs[i] = dz_i

            if i == 0:
                dw_i = np.dot(X, self.dzs[i].T)
                db_i = np.sum(self.dzs[i], axis=1, keepdims=True)
            else:
                dw_i = np.dot(self.activations[i - 1], self.dzs[i].T)
                db_i = np.sum(self.dzs[i])

            self.dws[i] = dw_i
            self.dbs[i] = db_i

    def geterror(self, x, y):
        ypred = self.predict(x)
        return MSE(y, ypred)

    def fit(self, x_train, y_train):
        lamda = 2

        for _ in range(self.epochs):
            for i in range(len(x_train)):
                self.forward_propagate(x_train[i][:, np.newaxis])
                self.back_propagate(x_train[i][:, np.newaxis], y_train[i])

                for i in range(len(self.weights)):
                    self.weights[i] = self.weights[i] - (1. / (len(x_train))) * self.lr * self.dws[i]
                    self.bias[i] = self.bias[i] - (1. / (len(x_train))) * self.lr * self.dbs[i]

            er = self.geterror(x_train, y_train)
            self.error.append(er)

    def predict(self, x):
        ypred = []
        for i in range(len(x)):
            acti = self.forward_propagate(x[i][:, np.newaxis])
            ypred.append(acti)

        # ypred = self.forward_propagate(x)
        # y_vals = [ypred[0][i] for i in range(len(ypred[0]))]

        return np.array([ypred[i][0][0] for i in range(len(ypred))])
