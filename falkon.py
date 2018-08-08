import numpy as np
from time import time
from progressbar import progressbar as bar
from numba import jit

from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from multiprocessing import cpu_count

from utility.kernel import gaussian


@jit(nopython=True)
def falkon(x_test, alpha, nystrom, gaussian_sigma):
    y_pred = np.zeros(shape=len(x_test), dtype=np.float32)

    for idx in range(len(y_pred)):
        for i in range(len(alpha)):
            y_pred[idx] += (alpha[i] * gaussian(x_test[idx], nystrom[i], gaussian_sigma))

    return y_pred


def train_falkon(x, y, m, gaussian_sigma, regularizer, max_iter):
    start = time()
    c = nystrom_centers(x, m)
    print("  --> Nystrom centers selected in {:.3f} seconds".format(time()-start))

    start = time()
    kmm = kernel_matrix(c, c, gaussian_sigma)
    print("  --> Kernel matrix based on centroids (KMM) computed in {:.3f} seconds".format(time() - start))

    start = time()
    rank = np.linalg.matrix_rank(M=kmm, hermitian=True)
    if rank != m:
        print("  --> Rank deficient KMM ({} instead of {})".format(rank, m))
    t = np.linalg.cholesky(a=(kmm + (1e-5*m*np.eye(m))))
    a = np.linalg.cholesky(a=(((t @ t.T)/m) + regularizer*np.eye(m)))
    print("  --> Computed T and A in {:.3f} seconds".format(time()-start))

    pool = PoolExecutor(max_workers=cpu_count())
    print("  --> Defined a Thread Pool with {} workers".format(pool._max_workers))

    start = time()
    b = np.linalg.solve(a, np.linalg.solve(t, kmn_vector(vec=y, train=x, nystrom=c, sigma=gaussian_sigma, pool=pool)))
    b /= len(y)
    print("  --> Computed b in {:.3f} seconds".format(time() - start))

    start = time()
    beta = conjgrad(lambda _beta: bhb(beta=_beta, a=a, t=t, train=x, nystrom=c, s=gaussian_sigma, lmb=regularizer,
                                      pool=pool), b=b, max_iter=max_iter)
    print("  --> Optimization done in {:.3f} seconds".format(time() - start))

    alpha = np.linalg.solve(t, np.linalg.solve(a, beta))

    return alpha, c


def nystrom_centers(x, m):
    c = x[np.random.choice(a=x.shape[0], size=m, replace=False), :]
    # d = np.diag(v=np.ones(shape=m))
    return c


@jit('float32[:, :](float32[:, :], float32[:, :], float32)', nopython=True)
def kernel_matrix(points1, points2, sigma):
    kernel_mtr = np.empty(shape=(len(points1), len(points2)), dtype=np.float32)

    for r in range(kernel_mtr.shape[0]):
        for c in range(kernel_mtr.shape[1]):
            kernel_mtr[r, c] = gaussian(points1[r], points2[c], sigma)

    return kernel_mtr


def kmn_knm_vector(vec, train, nystrom, sigma, pool):
    m = len(nystrom)

    res = np.zeros(shape=m, dtype=np.float32)
    for i in range(0, len(train), m*pool._max_workers):
        works = []
        for j in range(i, i + (m*pool._max_workers), m):
            subset_train = train[i:i + m, :]

            works.append(pool.submit(kernel_matrix, subset_train, nystrom, sigma))

        for w in works:
            subset_knm = w.result()
            res += (subset_knm.T @ (subset_knm @ vec))

    return res


def kmn_vector(vec, train, nystrom, sigma, pool):
    m = len(nystrom)

    res = np.zeros(shape=m, dtype=np.float32)
    for i in range(0, len(train), m*pool._max_workers):
        works = []
        subset_vecs = []
        for j in range(i, i + (m*pool._max_workers), m):
            subset_train = train[j:j + m, :]
            subset_vecs.append(j)

            works.append(pool.submit(kernel_matrix, nystrom, subset_train, sigma))

        for w, j in zip(works, subset_vecs):
            res += (w.result() @ vec[j:j + m])

    return res


def bhb(beta, a, t, train, nystrom, s, lmb, pool):
    _beta = kmn_knm_vector(vec=np.linalg.solve(t, np.linalg.solve(a, beta)),
                           train=train, nystrom=nystrom, sigma=s, pool=pool)
    return np.linalg.solve(a.T, np.linalg.solve(t.T, ((_beta / len(train)) + (lmb * np.linalg.solve(a, beta)))))


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
