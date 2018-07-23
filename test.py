import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from time import time

from falkon import falkon
from utility.kernel import linear, gaussian


def main(path, number_centroids, kernel_fun):
    # loading dataset as ndarray
    dataset = np.load(path)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1]-1))

    # adjusting label's range {-1, 1}
    dataset[:, 0] = 2*dataset[:, 0] - 1

    # defining dataset
    start = time()
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=0.20, random_state=91)
    print("Train and test set defined in {:.3f} seconds ({} train points)".format(time()-start, len(x_train)))

    # training falkon
    print("Starting falkon training routine...")
    start = time()
    alpha = falkon(x=x_train, y=y_train, m=number_centroids, kernel_function=kernel_fun, regularizer=1, max_iter=1000)
    print("Training finished in {:.3f} seconds".format(time()-start))

    print(alpha, alpha.shape)
    # testing falkon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("centroids", metavar='M', type=int, help='number of elements selected as Nystrom centroids')
    parser.add_argument("--kernel", type=str, default='linear', choices=['linear', 'gaussian'],
                        help='specify the kernel')
    parser.add_argument("--parameters", type=float, default=1,
                        help='define the parameters used for the kernel (c or sigma)')
    parser.add_argument("--gpu", type=bool, default=False, help='specify if the GPU is used (dafault = false)')

    args = parser.parse_args()

    ker = args.kernel
    if ker == 'linear':
        ker = lambda x, z: linear(x, z, args.parameters)
    elif ker == 'gaussian':
        ker = lambda x, z: gaussian(x, z, args.parameters)

    main(path=args.dataset, number_centroids=args.centroids, kernel_fun=ker)
