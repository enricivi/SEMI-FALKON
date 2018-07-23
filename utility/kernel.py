from numpy import inner


def linear(x, z, c):
    # data are listed for columns
    return inner(x, z) + c


def gaussian(x, z, s):
    pass
