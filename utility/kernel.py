import numpy as np
import cupy as cp


def gaussian(a, b, s):
    pass


def gpu_gaussian(a, b, s):
    km = cp.multiply(cp.matmul(a, b.T), -2)
    km += cp.power(a, 2).sum(axis=1).reshape(-1, 1)
    km += cp.power(b, 2).sum(axis=1)

    cp.multiply(km, -1 / (2 * (s ** 2)), out=km)
    cp.exp(km, out=km)
    return km