from numpy import vdot, exp
from numba import jit


@jit(nopython=True)
def linear(x, z, c):
    return vdot(x, z) + c


@jit(nopython=True)
def gaussian(x, z, s):
    gauss = x - z
    gauss = -vdot(gauss, gauss)
    gauss /= (2 * (s**2))
    return exp(gauss)
