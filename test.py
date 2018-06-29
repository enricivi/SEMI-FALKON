import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from utility.kernel import linear, gaussian


def main(path, centroids, kernel):
    # loading dataset as ndarray
    dataset = np.load(path)

    # adjusting label's range {-1, 1}
    dataset[:, 0] = 2*dataset[:, 0] - 1

    # defining dataset
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=0.20, random_state=91)

    # training falkon

    # testing falkon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for the test')
    parser.add_argument("centroids", metavar='M', type=int, help='number of elements used for the Nystrom method')
    parser.add_argument("--kernel", type=str, default='linear', choices=['linear', 'gaussian'],
                        help='specify the used kernel')
    parser.add_argument("--parameters", type=float, default=0,
                        help='define the parameters used for the kernel (c or sigma)')
    parser.add_argument("--gpu", type=bool, default=False, help='specify if the GPU is used (dafault = false)')

    args = parser.parse_args()

    ker = args.kernel
    if ker == 'linear':
        ker = lambda x, z: linear(x, z, args.parameters)
    elif ker == 'gaussian':
        ker = lambda x, z: gaussian(x, z, args.parameters)

    main(path=args.dataset, centroids=args.centroids, kernel=ker)
