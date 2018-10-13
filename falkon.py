import cupy as cp
import numpy as np
import psutil

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

    def fit(self, X, y):
        if self.gpu:
            self.memory_pool_ = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self.memory_pool_.malloc)

        self.random_state_ = check_random_state(self.random_state)
        self.nystrom_centers_ = self.upload(arr=X[self.random_state_.choice(a=X.shape[0], size=self.nystrom_length, replace=False), :])

        nystrom_kernels = self.__compute_kernels_matrix(self.nystrom_centers_, self.nystrom_centers_)

        self.__compute_a_t(kernels=nystrom_kernels)
        cp.cuda.Stream.null.synchronize()
        nystrom_kernels = self.__free_memory(nystrom_kernels)

        tmn = time()
        b = cp.linalg.solve(self.A_.T, cp.linalg.solve(self.T_.T, self.knm_dot_vec(X, arr=y/np.sqrt(X.shape[0]), transpose=True)))
        cp.cuda.Stream.null.synchronize()
        print(time() - tmn, self.download(b))

        #beta = self.__conjugate_gradient(w=lambda _beta: self.__compute_w(k, _beta), b=b)

        #self.weights_ = np.linalg.solve(self.T_, np.linalg.solve(self.A_, beta)) / np.sqrt(self.nystrom_length)

        return self

    def predict(self, X):
        y_pred = np.empty(shape=X.shape[0], dtype=np.float64)

        start_ = 0
        print("  predict progress: {:.2f} %".format((start_ / X.shape[0]) * 100), end='\r')
        while start_ < X.shape[0]:
            n_points = self.__fill_memory(start=start_, data_length=X.shape[0], dtype=X.dtype)

            tmp_kernel_matrix = self.__compute_kernels_matrix(X[start_:start_+n_points, :], self.nystrom_centers_)

            y_pred[start_:start_+n_points] = np.sum(a=tmp_kernel_matrix * self.weights_, axis=1)

            tmp_kernel_matrix = None  # cleaning memory

            start_ += n_points
            print("  predict progress: {:.2f} %".format((start_ / X.shape[0]) * 100), end='\r')
        print('')

        return y_pred

    # support functions

    def upload(self, arr):
        if self.gpu:
            return cp.asfortranarray(cp.asarray(a=arr))
        else:
            return arr

    def download(self, arr):
        if self.gpu:
            return arr.get()
        else:
            return arr

    def __compute_kernels_matrix(self, points1, points2):
        if self.gpu:
            block = (32, 32, 1)
            grid = ((len(points1) + (block[0] - 1)) // block[0], (len(points2) + (block[1] - 1)) // block[1])
            out = self.__compute_kernels_matrix_on_gpu(points1, points2, block=block, grid=grid)
            return out
        else:
            return self.__compute_kernels_matrix_on_cpu(points1, points2)

    def __compute_kernels_matrix_on_gpu(self, points1_gpu, points2_gpu, block, grid, out_gpu=None):
        nystrom_kernels = cp.empty(shape=(points1_gpu.shape[0], points2_gpu.shape[0]), dtype=points1_gpu.dtype, order='F') if out_gpu is None else out_gpu
        self.gauss_kernel(grid=grid, block=block, shared_mem=block[0]*self.nystrom_centers_.shape[1]*4,
                          args=(points1_gpu, points2_gpu, nystrom_kernels, nystrom_kernels.shape[0], nystrom_kernels.shape[1],
                                self.nystrom_centers_.shape[1], cp.float32(self.kernel_param))
                          )

        return nystrom_kernels

    def __compute_kernels_matrix_on_cpu(self, points1, points2):
        nystrom_kernels = np.empty(shape=(len(points1), len(points2)), dtype=np.float64)

        for idx in range(nystrom_kernels.shape[0]):
            nystrom_kernels[idx, :] = self.kernel_fun(points1[idx], points2)

        return nystrom_kernels

    def __compute_a_t(self, kernels):
        eye = np.eye(self.nystrom_length)
        if self.gpu:
            self.T_ = cp.linalg.cholesky(a=kernels + self.upload(np.finfo(kernels.dtype).eps * self.nystrom_length * eye)).T
            self.A_ = cp.linalg.cholesky(a=cp.divide((self.T_ @ self.T_.T), self.nystrom_length) + self.upload(self.gamma * eye)).T
        else:
            self.T_ = np.linalg.cholesky(a=kernels + (np.finfo(kernels.dtype).eps * self.nystrom_length * eye)).T
            self.A_ = np.linalg.cholesky(a=np.divide(self.T_ @ self.T_.T, self.nystrom_length) + (self.gamma * eye)).T

    def knm_dot_vec(self, x,  arr, transpose):
        xp = cp if self.gpu else np

        out = xp.zeros(shape=self.nystrom_length, dtype=arr.dtype)
        n_points = self.__fill_memory(start=0, data_length=arr.shape[0], dtype=arr.dtype)
        k = None
        for idx in range(0, arr.shape[0], n_points):
            if transpose:
                k = self.__compute_kernels_matrix(self.nystrom_centers_, self.upload(arr=x[idx:idx + n_points, :]))
            else:
                k = self.__compute_kernels_matrix(self.upload(arr=x[idx:idx + n_points, :]), self.nystrom_centers_)
            out = xp.add(out, xp.matmul(k, self.upload(arr[idx:idx + n_points])))

            k = self.__free_memory(k)

        return out

    def __fill_memory(self, start, data_length, dtype):
        n_points = 0
        available_memory = 0.0
        if self.gpu:
            available_memory = 5000000000 * self.memory_fraction
            n_points = int(min(available_memory / (self.nystrom_length * dtype.itemsize), data_length - start))
        else:
            available_memory = psutil.virtual_memory().available * self.memory_fraction
            n_points = int(min(available_memory / (self.nystrom_length * dtype.itemsize * 2), data_length - start))
        return n_points

    @staticmethod
    def __free_memory(arr):
        del arr
        return None

    def __conjugate_gradient(self, w, b):
        beta = np.zeros(shape=self.nystrom_length)

        r = b
        p = r
        rs_old = np.inner(r, r)

        for _ in range(self.optimizer_max_iter):
            wp = w(p)
            alpha = rs_old / np.inner(p, wp)

            beta += (alpha * p)
            r -= (alpha * wp)

            rs_new = np.inner(r, r)

            p = r + ((rs_new / rs_old) * p)
            rs_old = rs_new

        return beta
