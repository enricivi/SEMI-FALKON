import argparse
import numpy as np
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from time import time

# from falkon import falkon, train_falkon
from falkon import Falkon
from utility.kernel import *


def main(path, number_centroids, lmb, kernel, max_iter, xp):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float64)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1] - 1))

    # adjusting label's range {-1, 1}
    dataset[:, 0] = (2 * dataset[:, 0]) - 1

    # defining train and test set
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=0.2, random_state=7)
    dataset = None
    print("Train and test set defined")

    # removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("Standardization done")

    # training falkon
    print("Starting falkon training routine...")
    start = time()
    falkon = Falkon(nystrom_length=number_centroids, gamma=lmb, kernel_fun=kernel)
    falkon.fit(x_train, y_train)
    print("Training finished in {:.3f} seconds".format(time() - start))

    # testing falkon
    print("Starting falkon testing routine...")
    y_pred = np.sign(falkon.predict(x_test))
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    acc = np.sum(y_test == y_pred) / len(y_test)
    print("F1 score: {:.3f}".format(f1))
    print("Accuracy: {:.3f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("centroids", metavar='M', type=int, help='number of elements selected as Nystrom centroids')
    parser.add_argument("krr_lambda", metavar='L', type=float, help='ridge regression multiplier')
    parser.add_argument("--kernel", type=str, default='linear', choices=['linear', 'gaussian'],
                        help='specify the kernel')
    parser.add_argument("--max_iterations", type=int, default=500,
                        help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--ker_parameter", type=float, default=1,
                        help='define the parameters used for the kernel (c or sigma)')
    parser.add_argument("--gpu", type=bool, default=False, help='specify if the GPU is used (dafault = false)')

    args = parser.parse_args()

    kernel = None
    if args.kernel == "gaussian":
        kernel = lambda x, z: gaussian(x, z, args.ker_parameter)
    else:
        kernel = lambda x, z: linear(x, z, args.ker_parameter)
    xp = np if args.gpu else cp
    main(path=args.dataset, number_centroids=args.centroids, lmb=args.krr_lambda, kernel=kernel,
         max_iter=args.max_iterations, xp=xp)
