import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV

from time import time

# from falkon import falkon, train_falkon
from falkon import Falkon
from utility.kernel import *


def main(path, semi_supervised, max_iterations, gpu):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float32)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1] - 1))

    # defining train and test set
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=51630, random_state=None)
    print("Train and test set defined")

    # removing some labels (if semi_supervised > 0)
    labels_removed = int(len(y_train) * semi_supervised)
    if labels_removed > 0:
        y_train[np.random.choice(len(y_train), labels_removed, replace=False)] = 0
        print("{} labels removed".format(labels_removed))

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

    # hyperparameters tuninig
    print("Starting grid search...")
    falkon = Falkon(nystrom_length=None, gamma=None, kernel_fun='gaussian', kernel_param=None, optimizer_max_iter=max_iterations, gpu=gpu)
    parameters = {'nystrom_length': [10000, ], 'gamma': [1e-4, 1e-2, 1e-6], 'kernel_param': [4, 3, 6, 7]}
    gsht = GridSearchCV(falkon, param_grid=parameters, scoring=make_scorer(lambda true, pred: mean_squared_error(y_scaler.inverse_transform(true.reshape(-1, 1)), y_scaler.inverse_transform(pred.reshape(-1, 1)))), cv=3, verbose=3)
    gsht.fit(x_train, y_train)

    # printing some information of the best model
    print("Best model information: {} params, {:.3f} time (sec)".format(gsht.best_params_, gsht.refit_time_))

    # testing falkon
    print("Starting falkon testing routine...")
    y_pred = gsht.predict(x_test)
    mse = mean_squared_error(y_scaler.inverse_transform(y_test.reshape(-1, 1)), y_scaler.inverse_transform(y_pred.reshape(-1, 1)))
    print("Mean squared error: {:.3f}".format(mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("--semi_supervised", metavar='ss', type=float, default=0., help='percentage of elements [0, 1] to remove the label')
    parser.add_argument("--max_iterations", type=int, default=20, help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--gpu", type=bool, default=False, help='enable the GPU')

    args = parser.parse_args()

    main(path=args.dataset, semi_supervised=args.semi_supervised, max_iterations=args.max_iterations, gpu=args.gpu)