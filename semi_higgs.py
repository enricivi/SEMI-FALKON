import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from time import time

from falkon import Falkon
from utility.kernel import *


def labelling(y_pred, balance_constraint, lam0, theta0, max_violation, max_iterations=200):
    lam = lam0
    theta = theta0

    idx = None
    best_labels, y_l, y_u = None, None, None
    for idx in range(max_iterations):
        best_labels = get_best_labels(y_pred, lam)
        violation = np.sum(best_labels) - (y_pred.shape[0]*balance_constraint)
        if abs(violation) < max_violation:
            break
        elif violation < 0:
            y_l = best_labels
        else:
            y_u = best_labels

        if (y_l is not None) and (y_u is not None):
            # plane intersection
            numerator = np.sum(np.power(y_pred - y_l, 2)) - np.sum(np.power(y_pred - y_u, 2))
            denominator = np.sum(y_u) - np.sum(y_l)
            lam = numerator / denominator
        else:
            lam = lam + (theta * violation)
            theta = theta * 0.9

    return best_labels, lam, idx


def get_best_labels(functional_margin, lam):
    positive_labels = np.power(functional_margin - 1, 2) + lam
    negative_labels = np.power(functional_margin + 1, 2) - lam
    return 2.0*(negative_labels > positive_labels) - 1


def main(path, n_labeled, kernel_function, max_iterations, gpu):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float32)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1] - 1))

    # adjusting label's range {-1, 1}
    dataset[:, 0] = (2 * dataset[:, 0]) - 1

    # defining train and test set
    x_train, x_test, y_train, y_test = train_test_split(dataset[:, 1:], dataset[:, 0], test_size=100000, random_state=None)
    print("Train and test set defined (test: {} + , train: {} +, {} -)".format(np.sum(y_test == 1.), np.sum(y_train == 1.), np.sum(y_train == -1.)))

    # removing the mean and scaling to unit variance
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("Standardization done")

    # defining labeled and unlabeled set
    labeled = np.random.choice(np.where(y_train == 1)[0], size=int(n_labeled/2), replace=False)
    labeled = np.concatenate((labeled, np.random.choice(np.where(y_train == -1)[0], size=int(n_labeled/2), replace=False)), axis=0)
    x_labeled = x_train[labeled, :].copy()  # train
    y_labeled = y_train[labeled].copy()  # train
    x_train = np.delete(x_train, obj=labeled, axis=0)
    y_train = np.delete(y_train, obj=labeled)

    x_unlabeled = x_train.copy()  # test
    y_unlabeled = y_train.copy()  # test

    # choosing kernel function
    kernel = Kernel(kernel_function=kernel_function, gpu=gpu)

    # fitting falkon (semi-supervised scenario)
    best_score = -np.infty
    best_gamma, best_ker_param = None, None
    print("First training...")
    for gamma in [1e-8]:
        for ker_param in [5]:
            falkon = Falkon(nystrom_length=x_labeled.shape[0], gamma=gamma, kernel_fun=kernel.get_kernel(), kernel_param=ker_param, optimizer_max_iter=max_iterations, gpu=gpu)
            falkon.fit(x_labeled,  y_labeled)
            score = accuracy_score(y_labeled, np.sign(falkon.predict(x_labeled)))
            best_score, best_gamma, best_ker_param = (score, gamma, ker_param) if (score > best_score) else (best_score, best_gamma, best_ker_param)

    print("  -> [debug info] best score {:.3f} -- best gamma {} -- best kernel_param {}".format(best_score, best_gamma, best_ker_param))
    falkon = Falkon(nystrom_length=x_labeled.shape[0], gamma=best_gamma, kernel_fun=kernel.get_kernel(), kernel_param=best_ker_param, optimizer_max_iter=max_iterations, gpu=gpu)
    falkon.fit(x_labeled, y_labeled)
    functional_margin = falkon.predict(x_test)

    print(np.sum(functional_margin >= 0), np.sum(functional_margin < 0))

    print("Starting falkon testing routine...")
    accuracy = accuracy_score(y_test, np.sign(functional_margin))
    auc = roc_auc_score(y_test, functional_margin)
    print("Accuracy: {:.3f} - AUC: {:.3f}".format(accuracy, auc))

    print("Annealing loop...")
    functional_margin = falkon.predict(x_unlabeled)
    falkon = Falkon(nystrom_length=20000, gamma=best_gamma, kernel_fun=kernel.get_kernel(), kernel_param=best_ker_param, optimizer_max_iter=max_iterations, gpu=gpu)
    balance_constraint = (2 * 0.5) - 1  # 2r - 1
    start_ = time()
    for idx, weight in enumerate([0.1, 0.25, 0.5, 1.]):
        print(" -> iteration {}".format(idx+1))

        lam0 = ((2/x_unlabeled.shape[0])*np.sum(functional_margin)) - (2*balance_constraint)
        y_u, lam, iter = labelling(functional_margin, balance_constraint, lam0, 1., int(x_unlabeled.shape[0]*0.005))
        print("  -> [debug info] balance constraint {:.2}".format(np.divide(np.sum(y_u), x_unlabeled.shape[0])))
        print("  -> [debug info] lambda from {:.3e} to {:.3e} in {} iterations".format(lam0, lam, iter+1))
        print("  -> [debug info] wrong labels {}".format(np.sum(y_u != y_unlabeled)))

        sample_weights = ([1.] * x_labeled.shape[0]) + ([weight] * x_unlabeled.shape[0])
        falkon.fit(np.vstack((x_labeled, x_unlabeled)), np.concatenate((y_labeled, y_u)).astype(np.float32), sample_weights=sample_weights)
        functional_margin = falkon.predict(x_unlabeled)
    print("Annealing done in {:.3} seconds".format(time()-start_))

    # testing semi-supervised falkon
    print("Starting falkon testing routine...")
    functional_margin = falkon.predict(x_test)
    accuracy = accuracy_score(y_test, np.sign(functional_margin))
    auc = roc_auc_score(y_test, functional_margin)
    print("Accuracy: {:.3f} - AUC: {:.3f}".format(accuracy, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("--kernel", metavar='ker', type=str, default='gaussian', help='choose the kernel function')
    parser.add_argument("--n_labeled", metavar='ns', type=int, default=2, help='number of elements with label used (must be a multiple of 2)')
    parser.add_argument("--max_iterations", type=int, default=20, help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--gpu", type=bool, default=False, help='enable the GPU')

    args = parser.parse_args()

    main(path=args.dataset, kernel_function=args.kernel, n_labeled=args.n_labeled, max_iterations=args.max_iterations, gpu=args.gpu)