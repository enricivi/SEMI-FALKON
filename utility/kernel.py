from numpy import dot


def linear(x, z, c):
    # data are listed for columns
    return dot(x, z.T) + c


def gaussian(x, z, s):
    pass
