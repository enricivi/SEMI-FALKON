from numpy import inner, exp


def linear(x, z, c):
    return inner(x, z) + c


def gaussian(x, z, s):
    gauss = x - z
    gauss = -inner(gauss, gauss)
    gauss /= (2 * (s**2))
    return exp(gauss)
