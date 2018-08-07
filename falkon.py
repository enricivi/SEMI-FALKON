import numpy as np
from time import time
from progressbar import progressbar as bar
from numba import jit

from utility.kernel import gaussian


@jit(nopython=True)
def falkon(x_test, alpha, nystrom, gaussian_sigma):
    y_pred = np.zeros(shape=len(x_test), dtype=np.float32)

    for idx in range(len(y_pred)):
        for i in range(len(alpha)):
            y_pred[idx] += (alpha[i] * gaussian(x_test[idx], nystrom[i], gaussian_sigma))

    return y_pred


def train_falkon(x, y, m, gaussian_sigma, regularizer, max_iter):
    """
    :param x: is a matrix (size: n, features length) containing the train set
    :param y: is an array of label
    :param m: number of Nystrom centers (only uniform sampling)
    :param kernel_function: define the kernel used during the computation
    :param regularizer: lambda in objective function (see papers for more details)
    :param max_iter: maximum iterations number during the optimization
    :return: a vector of coefficients (size: m, 1) used for the future predictions
    """
    start = time()
    c, d = nystrom_centers(x, m)
    print("  --> Nystrom centers selected in {:.3f} seconds".format(time()-start))

    start = time()
    kmm = kernel_matrix(c, c, gaussian_sigma)
    print("  --> Kernel matrix based on centroids (KMM) computed in {:.3f} seconds".format(time() - start))

    start = time()
    rank = np.linalg.matrix_rank(M=kmm, hermitian=True)
    if rank != m:
        print("  --> Rank deficient KMM ({} instead of {})".format(rank, m))
    t = np.linalg.cholesky(a=((d @ kmm @ d) + (1e-5*m*np.eye(m))))  # 1e-5*m*eye is necessary because of numerical errors
    a = np.linalg.cholesky(a=(((t @ t.T)/m) + regularizer*np.eye(m)))
    print("  --> Computed T and A in {:.3f} seconds".format(time()-start))

    start = time()
    vec, kmn_knm = knm(vector=y, train=x, centroids=c, sigma=gaussian_sigma)
    print("  --> Computed KNM matrices in {:.3f} seconds".format(time() - start))

    start = time()
    b = np.linalg.solve(a, np.linalg.solve(t, d @ vec)) / len(y)
    print("  --> Computed b in {:.3f} seconds".format(time() - start))

    start = time()
    beta = conjgrad(lambda _beta: bhb(beta=_beta, a=a, t=t, d=d, kmn_knm=kmn_knm/len(y), kmm=regularizer*kmm),
                    b=b, max_iter=max_iter)
    print("  --> Optimization done in {:.3f} seconds".format(time() - start))

    alpha = d @ np.linalg.solve(t, np.linalg.solve(a, beta))

    return alpha, c


def nystrom_centers(x, m):
    c = x[np.random.choice(a=x.shape[0], size=m, replace=False), :]
    d = np.diag(v=np.ones(shape=m))
    return c, d


@jit(nopython=True)
def kernel_matrix(points1, points2, sigma):
    kernel_mtr = np.empty(shape=(len(points1), len(points2)), dtype=np.float32)

    for r in range(kernel_mtr.shape[0]):
        for c in range(kernel_mtr.shape[1]):
            kernel_mtr[r, c] = gaussian(points1[r], points2[c], sigma)

    return kernel_mtr


@jit(nopython=True)
def knm(vector, train, centroids, sigma):
    # computes KNM.T times KNM and KNM.T times vector
    m = len(centroids)
    n = len(train)

    vec = np.zeros(shape=m, dtype=np.float32)
    kmn_knm = np.zeros(shape=(m, m), dtype=np.float32)
    for i in range(0, n, m):
        subset_train = train[i:i + m, :]
        subset_vector = vector[i:i + m]

        tmp_kernel_matrix = kernel_matrix(centroids, subset_train, sigma)

        vec += tmp_kernel_matrix @ subset_vector
        kmn_knm += tmp_kernel_matrix @ tmp_kernel_matrix.T

    return vec, kmn_knm


def bhb(beta, a, t, d, kmn_knm, kmm):
    w = (kmn_knm + kmm) @ d @ np.linalg.solve(t, np.linalg.solve(a, beta))
    return np.linalg.solve(a.T, np.linalg.solve(t.T, d @ w))


def conjgrad(fun_w, b, max_iter):
    beta = np.zeros(shape=len(b), dtype=np.float32)

    r = b
    p = r
    rsold = np.inner(r, r)

    for _ in bar(range(max_iter)):
        wp = fun_w(p)
        alpha = rsold / np.inner(p, wp)

        beta += (alpha * p)
        r -= (alpha * wp)

        rsnew = np.inner(r, r)
        if np.sqrt(rsnew) < 1e-7:
            print("  --> stop criterion verified")
            break

        p = r + ((rsnew / rsold) * p)
        rsold = rsnew

    return beta
