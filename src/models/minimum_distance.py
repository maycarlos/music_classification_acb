import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator


class MDC(BaseEstimator):
    def __init__(self):
        self.class_list = None
        self.means = None

    def fit(self, X, y):
        self.class_list = np.unique(y, axis=0)

        self.means = np.zeros((len(self.class_list), X.shape[1]))

        for i in range(len(self.class_list)):
            temp = np.where(y == self.class_list[i])[0]

            self.means[i, :] = np.mean(X.iloc[temp, :], axis=0)

    def predict(self, X):
        temp = np.argmin(cdist(X, self.means), axis=1)
        y_pred = np.array([self.class_list[i] for i in temp])

        return y_pred

if __name__ == "__main__":
    ahah = MDC()

