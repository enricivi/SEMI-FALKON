import numpy as np

import psutil

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from time import time


class Falkon(BaseEstimator):
    def __init__(self, nystrom_length, gamma, kernel_fun, optimizer_max_iter=20, gpu=False, memory_fraction=0.90, random_state=None):
        self.nystrom_length = nystrom_length
        self.gamma = gamma
        self.kernel_fun = kernel_fun
        self.optimizer_max_iter = optimizer_max_iter
        self.gpu = gpu
        self.memory_fraction = memory_fraction
        self.random_state = random_state

        # Evaluated parameters
        self.random_state_ = None
        self.nystrom_centers_ = None
        self.T_ = None
        self.A_ = None
        self.weights_ = None

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        self.nystrom_centers_ = X[self.random_state_.choice(a=X.shape[0], size=self.nystrom_length, replace=False), :]

        nystrom_kernels = self.__compute_kernels_matrix(self.nystrom_centers_, self.nystrom_centers_)

        epsilon = np.eye(self.nystrom_length)
        self.T_ = np.linalg.cholesky(a=nystrom_kernels + (np.finfo(nystrom_kernels.dtype).eps * self.nystrom_length * epsilon)).T
        self.A_ = np.linalg.cholesky(a=np.divide(self.T_ @ self.T_.T, self.nystrom_length) + (self.gamma * epsilon)).T

        k, b = self.__compute_k_and_incomplete_b(X, y)
        b = (np.linalg.solve(self.A_.T, np.linalg.solve(self.T_.T, b)) / np.sqrt(self.nystrom_length))

        beta = self.__conjugate_gradient(w=lambda _beta: self.__compute_w(k, _beta), b=b)

        self.weights_ = np.linalg.solve(self.T_, np.linalg.solve(self.A_, beta)) / np.sqrt(self.nystrom_length)

        return self

    def predict(self, X):
        y_pred = np.empty(shape=X.shape[0], dtype=np.float64)

        start_ = 0
        print("  predict progress: {:.2f} %".format((start_ / X.shape[0]) * 100), end='\r')
        while start_ < X.shape[0]:
            n_points = self.__fill_memory(start=start_, data_length=X.shape[0], dtype=y_pred.dtype)

            tmp_kernel_matrix = self.__compute_kernels_matrix(X[start_:start_+n_points, :], self.nystrom_centers_)

            y_pred[start_:start_+n_points] = np.sum(a=tmp_kernel_matrix * self.weights_, axis=1)

            tmp_kernel_matrix = None  # cleaning memory

            start_ += n_points
            print("  predict progress: {:.2f} %".format((start_ / X.shape[0]) * 100), end='\r')
        print('')

        return y_pred

    def __compute_kernels_matrix(self, points1, points2):
        nystrom_kernels = np.empty(shape=(len(points1), len(points2)), dtype=np.float64)

        for idx in range(nystrom_kernels.shape[0]):
            nystrom_kernels[idx, :] = self.kernel_fun(points1[idx], points2)

        return nystrom_kernels

    def __compute_k_and_incomplete_b(self, X, y):
        k = np.zeros(shape=(self.nystrom_length, self.nystrom_length), dtype=np.float64)
        b = np.zeros(shape=self.nystrom_length, dtype=np.float64)

        start_ = 0
        print("  fit progress: {:.2f} %".format((start_ / X.shape[0]) * 100), end='\r')
        while start_ < X.shape[0]:
            n_points = self.__fill_memory(start=start_, data_length=X.shape[0], dtype=k.dtype)

            tmp_kernels_matrix_t = self.__compute_kernels_matrix(self.nystrom_centers_, X[start_:start_+n_points, :])

            k += (tmp_kernels_matrix_t @ tmp_kernels_matrix_t.T)
            b += (tmp_kernels_matrix_t @ y[start_:start_+n_points])

            tmp_kernels_matrix_t = None  # cleaning memory

            start_ += n_points
            print("  fit progress: {:.2f} %".format((start_ / X.shape[0]) * 100), end='\r')
        print('')

        return k, b

    def __compute_w(self, K, beta):
        zeta = np.linalg.solve(self.A_, beta)

        bhb_beta = self.gamma * zeta
        bhb_beta += (np.linalg.solve(self.T_.T, K @ np.linalg.solve(self.T_, zeta)) / self.nystrom_length)
        bhb_beta = np.linalg.solve(self.A_.T, bhb_beta)

        return bhb_beta

    def __fill_memory(self, start, data_length, dtype):
        available_memory = psutil.virtual_memory().available * self.memory_fraction
        return int(min(available_memory / (self.nystrom_length * dtype.itemsize * 2), data_length - start))

    def __conjugate_gradient(self, w, b):
        beta = np.zeros(shape=self.nystrom_length)

        r = b
        p = r
        rs_old = np.inner(r, r)

        for _ in range(self.optimizer_max_iter):
            wp = w(p)
            alpha = rs_old / np.inner(p, wp)

            beta += (alpha * p)
            r -= (alpha * wp)

            rs_new = np.inner(r, r)

            p = r + ((rs_new / rs_old) * p)
            rs_old = rs_new

        return beta
