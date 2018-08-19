import numpy as np
from time import time
from progressbar import progressbar as bar

from concurrent.futures import ProcessPoolExecutor as PoolExecutor


def falkon(x_test, alpha, nystrom, kernel):
    m = len(alpha)

    y_pred = np.empty(shape=len(x_test))
    for i in bar(range(0, len(x_test), m)):
        y_pred[i: i + m] = np.sum(kernel_matrix(x_test[i: i + m], nystrom, kernel) * alpha, axis=1)

    return y_pred


def train_falkon(x, y, m, kernel, lmb, max_iter, xp):
    start = time()
    c = nystrom_centers(x, m)
    print("  --> Nystrom centers selected in {:.3f} seconds".format(time()-start))

    start = time()
    kmm = kernel_matrix(c, c, kernel)
    print("  --> Kernel matrix based on centroids (KMM) computed in {:.3f} seconds".format(time() - start))

    start = time()
    t = np.linalg.cholesky(a=(kmm + (np.finfo(kmm.dtype).eps*m*np.eye(m)))).T
    a = np.linalg.cholesky(a=(((t @ t.T)/m) + lmb*np.eye(m))).T
    print("  --> Computed T and A in {:.3f} seconds".format(time()-start))

    workers = int(15000000000/(np.power(m, 2)*64))
    workers = workers if workers > 0 else 1
    pool = PoolExecutor(max_workers=workers)
    print("  --> Defined a Process Pool with {} workers".format(pool._max_workers))

    start = time()
    b = np.linalg.solve(a.T, np.linalg.solve(t.T, kmn_vector(vec=y, train=x, nystrom=c, kernel=kernel, pool=pool)))
    b /= len(y)
    print("  --> Computed b in {:.3f} seconds".format(time() - start))

    start = time()
    beta = conjgrad(lambda _beta: bhb(beta=_beta, a=a, t=t, train=x, nystrom=c, kernel=kernel, lmb=lmb, pool=pool),
                    b=b, max_iter=max_iter)
    print("  --> Optimization done in {:.3f} seconds".format(time() - start))

    alpha = np.linalg.solve(t, np.linalg.solve(a, beta))

    return alpha, c


def nystrom_centers(x, m):
    c = x[np.random.choice(a=x.shape[0], size=m, replace=False), :]
    return c


def kernel_matrix(points1, points2, kernel):
    kernel_mtr = np.empty(shape=(len(points1), len(points2)))

    for i in range(kernel_mtr.shape[0]):
        kernel_mtr[i, :] = kernel(points1[i], points2)

    return kernel_mtr


def kmn_knm_vector(vec, train, nystrom, kernel, pool):
    m = len(nystrom)

    res = np.zeros(shape=m)
    for i in range(0, len(train), m*pool._max_workers):
        works = []
        for j in range(i, i + (m*pool._max_workers), m):
            works.append(pool.submit(kernel_matrix, train[i:i + m, :], nystrom, kernel))

        for w in works:
            subset_knm = w.result()
            res += (subset_knm.T @ (subset_knm @ vec))

    return res


def kmn_vector(vec, train, nystrom, kernel, pool):
    m = len(nystrom)

    res = np.zeros(shape=m)
    for i in bar(range(0, len(train), m*pool._max_workers)):
        works = []
        subset_vecs = []
        for j in range(i, i + (m*pool._max_workers), m):
            subset_vecs.append(j)

            works.append(pool.submit(kernel_matrix, nystrom, train[j:j + m, :], kernel))

        for w, j in zip(works, subset_vecs):
            res += (w.result() @ vec[j:j + m])

    return res


def bhb(beta, a, t, train, nystrom, kernel, lmb, pool):
    z = np.linalg.solve(a, beta)
    _beta = kmn_knm_vector(vec=np.linalg.solve(t, z), train=train, nystrom=nystrom, kernel=kernel, pool=pool)
    return np.linalg.solve(a.T, np.linalg.solve(t.T, (_beta / len(train)) + (lmb * z)))


def conjgrad(fun_w, b, max_iter):
    beta = np.zeros(shape=len(b))

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
