import math

from numpy import exp, power
from numpy.linalg import norm
from numba import cuda


def linear(x, z, c):
    pass


def gaussian(x, z, s):
    gauss = power(norm(x=z - x, axis=1, ord=2), 2)
    gauss /= (-2 * (s**2))
    return exp(gauss)


@cuda.jit('float64(float64[:], float64[:], float64)', device=True)
def gpu_gaussian(x, z, s):
    gauss = 0.0
    for idx in range(len(x)):
        tmp = x[idx] - z[idx]
        gauss += (tmp * tmp)
    gauss /= (-2 * s * s)
    return math.exp(gauss)