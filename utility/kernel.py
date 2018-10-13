from numpy import exp, power
from numpy.linalg import norm


def linear(x, z, c):
    pass


def gaussian(x, z, s):
    gauss = power(norm(x=z - x, axis=1, ord=2), 2)
    gauss /= (-2 * (s**2))
    return exp(gauss)
