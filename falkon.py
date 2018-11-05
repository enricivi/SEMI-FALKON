import numpy as np
import cupy as cp
from cupy.cuda import cublas
from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
import cupy.cuda as cuda

import psutil
import GPUtil as gputil

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from time import time


class Falkon(BaseEstimator):
    def __init__(self, nystrom_length, gamma, kernel_fun, kernel_param, optimizer_max_iter=20, gpu=False, memory_fraction=0.90, random_state=None):
        self.nystrom_length = nystrom_length
        self.gamma = gamma
        self.kernel_fun = kernel_fun
        self.kernel_param = kernel_param
        self.optimizer_max_iter = optimizer_max_iter
        self.gpu = gpu
        self.memory_fraction = memory_fraction
        self.random_state = random_state

        # Evaluated parameters
        self.memory_pool_ = None
        self.random_state_ = None
        self.nystrom_centers_ = None
        self.T_ = None
        self.A_ = None
        self.weights_ = None

        # test
        """
        self.gauss_kernel = cp.RawKernel(r'''
            extern "C" __device__ float gauss(float *a, float *b, int i, int j, int len_a, int len_b, int nfeatures, float s) {	
                float val = 0.0;
                float diff;
                for (int idx = 0; idx < nfeatures; idx++) {
                    diff = a[i + (idx * len_a)] - b[j + (idx * len_b)];
                    val += (diff * diff);
                }
                val /= (-2 * s * s);
                return __expf(val);
            }
        
            extern "C" __global__ void gauss_kernel(float *a, float *b, float *o, int len_a, int len_b, int nfeatures, float s) {
                int row = threadIdx.x + (blockIdx.x * blockDim.x);            	
                int col = threadIdx.y + (blockIdx.y * blockDim.y);
                
                extern __shared__ float shared_a[];
                if ((row < len_a) && (threadIdx.y == 0)) {
                    for (int idx=0; idx < nfeatures; idx++) {
                        shared_a[threadIdx.x + (idx * blockDim.x)] = a[row + (idx * len_a)];
                    }
                }
                
                __syncthreads();      
            
                int pos;
                if ((row < len_a) && (col < len_b)) {
                    pos = (col * len_a) + row;
                    o[pos] = gauss(shared_a, b, threadIdx.x, col, blockDim.x, len_b, nfeatures, s);
                }
            }
        ''', 'gauss_kernel', ('--use_fast_math', ))
        """

    # train/predict functions

    def fit(self, X, y):
        if self.gpu:
            self.memory_pool_ = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self.memory_pool_.malloc)

        self.random_state_ = check_random_state(self.random_state)
        self.nystrom_centers_ = self.upload(arr=X[self.random_state_.choice(a=X.shape[0], size=self.nystrom_length, replace=False), :])

        nystrom_kernels = self.__compute_kernels_matrix(self.nystrom_centers_, self.nystrom_centers_)

        self.__compute_a_t(kernels=nystrom_kernels)
        nystrom_kernels = self.__free_memory(nystrom_kernels)

        b = self.__knm_prod(x=X, b=None, y=y/np.sqrt(X.shape[0]))
        cp_solve_triangular(self.T_, b, trans='T', overwrite_b=True)
        cp_solve_triangular(self.A_, b, trans='T', overwrite_b=True)

        beta = self.__conjugate_gradient(w=lambda _beta: self.__compute_php(_beta, X), b=b)

        cp_solve_triangular(self.A_, beta, overwrite_b=True)
        cp_solve_triangular(self.T_, beta, overwrite_b=True)
        self.weights_ = self.download(cp.divide(beta, np.sqrt(X.shape[0])))

        return self

    def predict(self, X):
        self.memory_pool_.free_all_blocks()
        y_pred = cp.empty(shape=X.shape[0], dtype=np.float32)

        w = self.upload(self.weights_)
        n_points = self.__fill_memory(start=0, data_length=X.shape[0], dtype=X.dtype)
        k = None
        for idx in range(0, X.shape[0], n_points):
            k = self.__compute_kernels_matrix(self.upload(X[idx:idx+n_points, :]), self.nystrom_centers_)
            y_pred[idx:idx+n_points] = cp.sum(a=cp.multiply(k, w), axis=1)
            k = self.__free_memory(k)

        return self.download(y_pred)

    # support functions

    def __compute_kernels_matrix(self, points1, points2):
        return self.kernel_fun(self.upload(points1), self.upload(points2), np.float32(self.kernel_param))

    def __compute_a_t(self, kernels):
        xp = cp if self.gpu else np

        eye = np.eye(self.nystrom_length, dtype=kernels.dtype)
        self.T_ = xp.linalg.cholesky(a=kernels + self.upload(np.finfo(kernels.dtype).eps * self.nystrom_length * eye)).T
        self.A_ = xp.divide(xp.matmul(self.T_, self.T_.T), self.nystrom_length)
        self.A_ = xp.linalg.cholesky(self.A_ + self.upload(self.gamma * eye)).T

        return

    def __compute_php(self, beta, x):
        ans = None
        if self.gpu:
            zeta = cp_solve_triangular(self.A_, beta)
            ans = cp_solve_triangular(self.T_, zeta)
            ans = self.__knm_prod(x=x, b=ans, y=None)
            cp_solve_triangular(self.T_, ans, trans='T', overwrite_b=True)
            cp.add(cp.divide(ans, x.shape[0], out=ans), cp.multiply(zeta, self.gamma, out=zeta), out=ans)
            cp_solve_triangular(self.A_, ans, trans='T', overwrite_b=True)

            cp.cuda.Stream.null.synchronize()
        else:
            raise NotImplementedError
        return ans

    def __knm_prod(self, x, b, y):
        xp = None
        handle = None
        if self.gpu:
            self.memory_pool_.free_all_blocks()
            xp = cp
            handle = cuda.device.get_cublas_handle()
        else:
            xp = np

        out = xp.zeros(shape=self.nystrom_length, dtype=x.dtype)
        n_points = self.__fill_memory(start=0, data_length=x.shape[0], dtype=x.dtype)
        b = self.upload(b) if y is None else None
        k = None
        for idx in range(0, x.shape[0], n_points):
            k = xp.asfortranarray(self.__compute_kernels_matrix(self.upload(arr=x[idx:idx + n_points, :]), self.nystrom_centers_))
            if y is None:
                kb = xp.empty(shape=k.shape[0], dtype=x.dtype)
                cublas.sgemv(handle, 0, k.shape[0], k.shape[1], 1.0, k.data.ptr, k.shape[0], b.data.ptr, 1, 0, kb.data.ptr, 1)
                cublas.sgemv(handle, 1, k.shape[0], k.shape[1], 1.0, k.data.ptr, k.shape[0], kb.data.ptr, 1, 1.0, out.data.ptr, 1)
                kb = self.__free_memory(kb)
            else:
                cublas.sgemv(handle, 1, k.shape[0], k.shape[1], 1.0, k.data.ptr, k.shape[0], self.upload(y[idx:idx + n_points]).data.ptr, 1, 1.0, out.data.ptr, 1)

            k = self.__free_memory(k)
        return out

    # memory function

    def upload(self, arr):
        return cp.asarray(a=arr) if self.gpu else arr

    def download(self, arr):
        return arr.get() if self.gpu else arr

    def __fill_memory(self, start, data_length, dtype):
        n_points = 0
        available_memory = 0.0
        if self.gpu:
            available_memory = gputil.getGPUs()[0].memoryFree * (1024 ** 2)
            available_memory = available_memory * self.memory_fraction
            n_points = int(min(available_memory / (self.nystrom_length * dtype.itemsize * 2), data_length - start))
        else:
            available_memory = psutil.virtual_memory().available * self.memory_fraction
            n_points = int(min(available_memory / (self.nystrom_length * dtype.itemsize * 2), data_length - start))
        return n_points

    @staticmethod
    def __free_memory(arr):
        del arr
        return None

    # optimization method

    def __conjugate_gradient(self, w, b):
        beta = np.zeros(shape=self.nystrom_length, dtype=np.float32)

        r = self.download(b)
        p = r.copy()
        rs_old = np.inner(r, r)

        for iteration in range(self.optimizer_max_iter):
            wp = self.download(w(self.upload(p)))
            alpha = rs_old / np.inner(p, wp)

            beta += (alpha * p)

            r -= (alpha * wp)
            rs_new = np.inner(r, r)
            p = r + ((rs_new / rs_old) * p)
            rs_old = rs_new

        return self.upload(beta)
