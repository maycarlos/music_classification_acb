import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_X_y


class MinimumDistanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, metric="euclidean"):
        self.classes_ = None
        self.__means = None
        self.metric = metric

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)
        self.__means = np.zeros((len(self.classes_), X.shape[1]))

        for i in range(len(self.classes_)):
            temp = np.where(y == self.classes_[i])[0]

            self.__means[i, :] = np.mean(X[temp], axis=0)

        return self

    def predict(self, X):
        check_is_fitted(self)
        temp = np.argmin(cdist(X, self.__means, metric=self.metric), axis=1)
        y_pred = np.array([self.classes_[i] for i in temp])

        return y_pred

    @property
    def class_means(self):
        return self.__means
