import numpy as np
from time import time
from progressbar import progressbar as bar


def falkon(x, y, m, kernel_function, regularizer, max_iter):
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
    kmm = kernel_matrix(c, c, kernel_function)
    print("  --> Kernel matrix based on centroids (KMM) computed in {:.3f} seconds".format(time() - start))

    kmm_rank = np.linalg.matrix_rank(M=kmm, hermitian=True)
    t = None
    q = None
    start = time()
    if kmm_rank == m:
        print("  --> Full rank KMM")
        t = np.linalg.cholesky(a=((d @ kmm @ d) + np.eye(m)))
    else:
        print("  --> Rank deficient KMM (rank {})".format(kmm_rank))
        q, _ = np.linalg.qr(a=(d @ kmm @ d))
        t = np.linalg.cholesky(a=((q.T @ d @ kmm @ d @ q) + np.eye(m)))
    a = np.linalg.cholesky(a=((t/m @ t.T) + regularizer*np.eye(m)))
    print("  --> Computed Q (if KMM has a deficient rank), T and A in {:.3f} seconds".format(time()-start))

    start = time()
    kmn_knm = kmn_times_knm(x, c, kernel_function)
    print("  --> Computed KNM.T times KNM in {:.3f} seconds".format(time() - start))

    start = time()
    r = np.linalg.solve(a, np.linalg.solve(t, q.T @ d @ kmn_times_vector(y, x, c, kernel_function)))
    print("  --> Computed B.T times Knm.T times y in {:.3f} seconds".format(time() - start))

    start = time()
    beta = conjgrad(lambda b: bhb(beta=b, a=a, t=t, d=d, kmn_knm=kmn_knm/len(x), kmm=regularizer*kmm, q=q), r, max_iter)
    print("  --> Optimization done in {:.3f} seconds".format(time() - start))

    alpha = None
    if q is None:
        alpha = np.linalg.solve(t, np.linalg.solve(a, beta))
    else:
        alpha = q @ np.linalg.solve(t, np.linalg.solve(a, beta))

    return d @ alpha


def nystrom_centers(x, m):
    c = x[np.random.choice(a=x.shape[0], size=m, replace=False), :]
    d = np.diag(v=np.ones(shape=m))

    return c, d


def kernel_matrix(points1, points2, fun):
    kernels = np.empty(shape=(len(points1), len(points2)), dtype=np.float32)

    for i in range(kernels.shape[0]):
        p = points1[i]
        for j in range(kernels.shape[1]):
            kernels[i, j] = fun(p, points2[j])

    return kernels


def kmn_times_vector(vector, train, centroids, fun):
    # TODO: RICONTROLLA!!
    m = len(centroids)
    n = len(train)

    product = np.zeros(shape=m, dtype=np.float32)
    for i in range(0, n, m):
        subset_train = train[i:i + m, :]
        subset_vector = vector[i:i + m]
        tmp_kernel_matrix = kernel_matrix(subset_train, centroids, fun)

        product += tmp_kernel_matrix.T @ subset_vector

    return product


def kmn_times_knm(train, centroids, fun):
    # TODO: controlla
    m = len(centroids)
    n = len(train)

    kmn_knm = np.zeros(shape=(m, m), dtype=np.float32)
    for i in range(0, n, m):
        subset_train = train[i:i+m, :]
        tmp_kernel_matrix = kernel_matrix(subset_train, centroids, fun)

        kmn_knm += tmp_kernel_matrix.T @ tmp_kernel_matrix

    return kmn_knm


def bhb(beta, a, t, d, kmn_knm, kmm, q):
    # TODO: controlla
    kmm = kmn_knm + kmm

    w = None
    if q is None:
        w = d @ kmm @ d @ np.linalg.solve(t, np.linalg.solve(a, beta))
    else:
        w = q.T @ d @ kmm @ d @ q @ np.linalg.solve(t, np.linalg.solve(a, beta))

    return np.linalg.solve(a.T, np.linalg.solve(t.T, w))


def conjgrad(fun, r, max_iter):
    beta = np.zeros(shape=len(r), dtype=np.float32)

    p = r
    rsold = np.inner(r, r)

    for _ in bar(range(max_iter)):
        ap = fun(p)
        a = rsold / np.inner(p, ap)
        beta = beta + (a * p)
        r = r - (a * ap)
        rsnew = np.inner(r, r)
        p = r + (rsnew/rsold) * p
        rsold = rsnew

    return beta
