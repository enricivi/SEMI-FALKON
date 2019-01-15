import argparse
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score

from matplotlib import pyplot as plt

from time import time

from falkon import Falkon
from utility.kernel import *


def plot_2d_dataset(dataset, unlabeled, y_pred, filepath="./fig.png"):
    print("saving figure '{}'".format(filepath))
    colors = np.asarray(['r' if (pred >= 0) else 'b' for pred in y_pred])
    plt.scatter(dataset[unlabeled, 1], dataset[unlabeled, 2], color=colors)
    plt.savefig(filepath)


def main(path, n_labeled, kernel_function, max_iterations, gpu):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float32)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1] - 1))

    # defining labeled and unlabeled set
    labeled = np.random.choice(np.where(dataset[:, 0] == 1)[0], size=int(n_labeled/2))
    labeled = np.concatenate((labeled, np.random.choice(np.where(dataset[:, 0] == -1)[0], size=int(n_labeled/2))), axis=0)
    unlabeled = np.delete(np.arange(start=0, stop=dataset.shape[0], step=1), obj=labeled)
    y_unlabelled = dataset[unlabeled, 0].copy()

    # choosing kernel function
    kernel = Kernel(kernel_function=kernel_function, gpu=gpu)

    # fitting falkon (semi-supervised scenario)
    print("Start training...")
    falkon = Falkon(nystrom_length=2, gamma=1e-6, kernel_fun=kernel.get_kernel(), kernel_param=4, optimizer_max_iter=max_iterations, gpu=gpu)
    falkon.fit(dataset[labeled, 1:], dataset[labeled, 0])
    plot_2d_dataset(dataset, unlabeled, falkon.predict(dataset[unlabeled, 1:]))

    

    # testing semi-supervised falkon
    print("Starting falkon testing routine...")
    y_pred = falkon.predict(dataset[unlabeled, 1:])
    accuracy = accuracy_score(y_unlabelled, np.sign(y_pred))
    auc = roc_auc_score(y_unlabelled, y_pred)
    print("Accuracy: {:.3f} - AUC: {:.3f}".format(accuracy, auc))

    plot_2d_dataset(dataset, unlabeled, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("--kernel", metavar='ker', type=str, default='gaussian', help='choose the kernel function')
    parser.add_argument("--n_labeled", metavar='ns', type=int, default=2, help='number of elements with label used (must be a multiple of 2)')
    parser.add_argument("--max_iterations", type=int, default=20, help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--gpu", type=bool, default=False, help='enable the GPU')

    args = parser.parse_args()

    main(path=args.dataset, kernel_function=args.kernel, n_labeled=args.n_labeled, max_iterations=args.max_iterations, gpu=args.gpu)