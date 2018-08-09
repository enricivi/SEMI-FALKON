import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from time import time

from falkon import falkon, train_falkon


def main(path, number_centroids, lmb, gauss_sigma, max_iter):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float64)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1]-1))

    # adjusting label's range {-1, 1}
    dataset[:, 0] = (2 * dataset[:, 0]) - 1

    # defining train and test set
    start = time()
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=0.2, random_state=7)
    dataset = None
    print("Train and test set defined in {:.3f} seconds ({} train points)".format(time()-start, len(x_train)))

    # removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # training falkon
    print("Starting falkon training routine...")
    start = time()
    alpha, nystrom = train_falkon(x=x_train, y=y_train, m=number_centroids, gaussian_sigma=gauss_sigma, regularizer=lmb,
                                  max_iter=max_iter)
    print("Training finished in {:.3f} seconds".format(time()-start))

    # testing falkon
    print("Starting falkon testing routine...")
    y_pred = np.sign(falkon(x_test=x_test, alpha=alpha, nystrom=nystrom, gaussian_sigma=gauss_sigma))
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    print("F1 score: {:.3f}".format(f1))
    print("Area Under ROC Curve score: {:.3f}".format(auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("centroids", metavar='M', type=int, help='number of elements selected as Nystrom centroids')
    parser.add_argument("regularizer", metavar='L', type=float, help='ridge regression multiplier')
    parser.add_argument("--kernel", type=str, default='linear', choices=['linear', 'gaussian'],
                        help='specify the kernel')
    parser.add_argument("--max_iterations", type=int, default=500,
                        help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--ker_parameter", type=float, default=1,
                        help='define the parameters used for the kernel (c or sigma)')
    parser.add_argument("--gpu", type=bool, default=False, help='specify if the GPU is used (dafault = false)')

    args = parser.parse_args()

    main(path=args.dataset, number_centroids=args.centroids, lmb=args.regularizer, gauss_sigma=args.ker_parameter,
         max_iter=args.max_iterations)
