import numpy as np


class DataLoader:
    def __init__(self):
        # X is of shape (N,F,T)
        # where N is the space dim (node number), F is the feature dim,
        # T is the time dim
        self.X = None

        # A is the adjacency matrix, of shape (N,N)
        self.A = None

    def train_test_split(self, dt=1, ratio=0.7):
        T = self.X.shape[-1]
        train_set = self.X[:, :, : int(T * ratio)]
        test_set = self.X[:, :, int(T * ratio) :]

        X_train, y_train, X_test, y_test = list(), list(), list(), list()
        for i in range(train_set.shape[-1] - dt):
            X = train_set[:, :, i : i + dt]
            y = train_set[:, :, i + dt : i + 2 * dt]
            X_train.append(X[np.newaxis, :, :, :])
            y_train.append(y[np.newaxis, :, :, :])

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        for i in range(test_set.shape[-1] - dt):
            X = test_set[:, :, i : i + dt]
            y = test_set[:, :, i + dt : i + 2 * dt]
            X_test.append(X[np.newaxis, :, :, :])
            y_test.append(y[np.newaxis, :, :, :])

        X_test = np.concatenate(X_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        return (
            np.squeeze(X_train),
            np.squeeze(y_train),
            np.squeeze(X_test),
            np.squeeze(y_test),
        )


class RingRoadLoader(DataLoader):
    def __init__(self, rho, V):
        # rho and u are of shape (N, T)
        # rho or may have one T dim
        super().__init__()

        rho = rho[:, np.newaxis, :]
        V = V[:, np.newaxis, :]

        # X is of shape (N,F,T), where F=2
        self.X = np.concatenate((rho, V), axis=1)

        # For a ringroad, the diagonal and off-diagonal elements are 1, and 0 elsewhere.
        N = self.X.shape[0]
        a = np.zeros((1, N))[0]
        b = np.ones((1, N - 1))[0]
        self.A = np.diag(a, 0) + np.diag(b, -1) + np.diag(b, 1)
        self.A[0, -1] = 1
        self.A[-1, 0] = 1
