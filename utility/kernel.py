import numpy as np
import cupy as cp

l2norm_pow2 = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    'l2norm_pow2'  # kernel name
)


def gaussian(a, b, s):
    pass


def gpu_gaussian(a, b, s):
    km = cp.multiply(cp.matmul(a, b.T), -2)
    km += cp.power(a, 2).sum(axis=1).reshape(-1, 1)
    km += cp.power(b, 2).sum(axis=1)
    # km += l2norm_pow2(a, axis=1).reshape(-1, 1)
    # km += l2norm_pow2(b, axis=1)

    cp.multiply(km, -1 / (2 * s * s), out=km)
    cp.exp(km, out=km)
    return km
