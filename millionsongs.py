import argparse
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from time import time

from falkon import Falkon
from utility.kernel import Kernel


def main(path, kernel_function, max_iterations, gpu):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float32)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1] - 1))

    # defining train and test set
    # x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=51630, random_state=None)
    x_train = dataset[0:463715, 1:]; x_test = dataset[463715:515345, 1:]
    y_train = dataset[0:463715, 0]; y_test = dataset[463715:515345, 0]
    print("Train and test set defined")

    # removing the mean and scaling to unit variance
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(x_train)
    y_scaler.fit(y_train.reshape(-1, 1))
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
    print("Standardization done")

    # choosing kernel function
    kernel = Kernel(kernel_function=kernel_function, gpu=gpu)

    # fitting falkon
    print("Starting falkon fit routine...")
    falkon = Falkon(nystrom_length=10000, gamma=1e-6, kernel_fun=kernel.get_kernel(), kernel_param=6, optimizer_max_iter=max_iterations, gpu=gpu)
    start_ = time()
    falkon.fit(x_train, y_train)
    print("Fitting time: {:.3f} seconds".format(time() - start_))

    # testing falkon
    print("Starting falkon testing routine...")
    y_pred = falkon.predict(x_test)
    mse = mean_squared_error(inv_transform(y_scaler, y_test), inv_transform(y_scaler, y_pred))
    print("Mean squared error: {:.3f}".format(mse))


def inv_transform(scaler, data):
    return scaler.inverse_transform(data.reshape(-1, 1)).reshape(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("--kernel", metavar='ker', type=str, default='gaussian', help='choose the kernel function')
    parser.add_argument("--max_iterations", type=int, default=20, help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--gpu", type=bool, default=False, help='enable the GPU')

    args = parser.parse_args()

    main(path=args.dataset, kernel_function=args.kernel, max_iterations=args.max_iterations, gpu=args.gpu)
