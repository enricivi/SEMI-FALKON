import numpy as np
import cupy as cp


class Kernel:
    def __init__(self, kernel_function, gpu):
        self.kernel = None

        if gpu:
            if kernel_function == 'gaussian':
                self.kernel = self.gpu_gaussian
        else:
            if kernel_function == 'gaussian':
                self.kernel = self.gaussian

    def get_kernel(self):
        return self.kernel

    def gaussian(self, a, b, s):
        km = np.empty(shape=(a.shape[0], b.shape[0]), dtype=a.dtype)
        km = np.multiply(np.dot(a, b.T, out=km), -2, out=km)
        km += np.power(a, 2).sum(axis=1).reshape(-1, 1)
        km += np.power(b, 2).sum(axis=1)

        np.multiply(km, -1 / (2 * s * s), out=km)
        np.exp(km, out=km)
        return km

    def gpu_gaussian(self, a, b, s):
        km = cp.empty(shape=(a.shape[0], b.shape[0]), dtype=a.dtype)
        km = cp.multiply(cp.dot(a, b.T, out=km), -2, out=km)
        km += cp.power(a, 2).sum(axis=1).reshape(-1, 1)
        km += cp.power(b, 2).sum(axis=1)

        cp.multiply(km, -1 / (2 * s * s), out=km)
        cp.exp(km, out=km)
        return km
