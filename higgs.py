import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from time import time

from falkon import Falkon
from utility.kernel import *


def main(path, kernel_function, max_iterations, gpu):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float32)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1] - 1))

    # adjusting label's range {-1, 1}
    dataset[:, 0] = (2 * dataset[:, 0]) - 1

    # defining train and test set
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=0.2, random_state=None)
    print("Train and test set defined (test: {} + , train: {} +, {} -)".format(np.sum(y_test == 1.), np.sum(y_train == 1.), np.sum(y_train == -1.)))

    # removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("Standardization done")

    # choosing kernel function
    kernel = Kernel(kernel_function=kernel_function, gpu=gpu)

    # fitting falkon
    print("Starting falkon fitting routine...")
    falkon = Falkon(nystrom_length=20000, gamma=1e-8, kernel_fun=kernel.get_kernel(), kernel_param=5, optimizer_max_iter=max_iterations, gpu=gpu)
    start_ = time()
    falkon.fit(x_train, y_train)
    print("Fitting time: {:.3f} seconds".format(time() - start_))

    # testing falkon
    print("Starting falkon testing routine...")
    y_pred = falkon.predict(x_test)
    accuracy = accuracy_score(y_test, np.sign(y_pred))
    auc = roc_auc_score(y_test, y_pred)
    print("Accuracy: {:.3f} - AUC: {:.3f}".format(accuracy, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("--kernel", metavar='ker', type=str, default='gaussian', help='choose the kernel function')
    parser.add_argument("--max_iterations", type=int, default=20, help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--gpu", type=bool, default=False, help='enable the GPU')

    args = parser.parse_args()

    main(path=args.dataset, kernel_function=args.kernel, max_iterations=args.max_iterations, gpu=args.gpu)