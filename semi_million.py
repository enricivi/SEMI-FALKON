import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from time import time
from falkon import Falkon
from utility.kernel import Kernel

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

def main(path, kernel_function, max_iterations, gpu):
    # loading dataset as ndarray
    dataset = np.load(path).astype(np.float32)
    print("Dataset loaded ({} points, {} features per point)".format(dataset.shape[0], dataset.shape[1] - 1))

    # defining train and test set
    x_train = dataset[0:463715, 1:]; x_test = dataset[463715:515345, 1:]
    y_train = dataset[0:463715, 0]; y_test = dataset[463715:515345, 0]
    print("Train and test set defined")

    # creating the unsupervised part of the dataset (using x_train)
    labeled_ids, unlabeled_ids = train_test_split(range(x_train.shape[0]), test_size=0.7, random_state=42)
    x_labeled, y_labeled = x_train[labeled_ids, :], y_train[labeled_ids]
    x_unlabeled, y_unlabeled = x_train[unlabeled_ids, :], y_train[unlabeled_ids]
    print("Labeled examples {}, Unlabeled examples {}".format(x_labeled.shape[0], x_unlabeled.shape[0]))
    
    # labels binarization (-1 from 1922 to 2002, 1 from 2002 to 2011) -- balanced (labeled) dataset
    y_labeled, y_test = (y_labeled >= 2002).astype(np.float32), (y_test >= 2002).astype(np.float32)
    y_unlabeled = (y_unlabeled >= 2002).astype(np.float32)
    y_labeled, y_unlabeled = (2 * y_labeled) - 1, (2 * y_unlabeled) - 1
    y_test = (2 * y_test) - 1

    # removing the mean and scaling to unit variance
    x_scaler = StandardScaler()
    x_scaler.fit(x_train) # using labeled + unlabeled part 
    x_labeled, x_unlabeled = x_scaler.transform(x_labeled), x_scaler.transform(x_unlabeled)
    x_test = x_scaler.transform(x_test)
    print("Standardization done")

    # choosing kernel function
    kernel = Kernel(kernel_function=kernel_function, gpu=gpu)
    
    # training
    print("First training...")
    falkon = Falkon(nystrom_length=round(np.sqrt(x_labeled.shape[0])), gamma=1e-6, kernel_fun=kernel.get_kernel(), kernel_param=6, optimizer_max_iter=max_iterations, gpu=gpu)
    falkon.fit(x_labeled, y_labeled)
    functional_margin = falkon.predict(x_test)
    
    # initial Accuracy, AUC_ROC
    accuracy = accuracy_score(y_test, np.sign(functional_margin))
    auc_roc = roc_auc_score(y_test, functional_margin)
    print("Accuracy: {:.4f} - AUC: {:.4f}".format(accuracy, auc_roc))

    print("Annealing loop...")
    functional_margin = falkon.predict(x_unlabeled)
    falkon = Falkon(nystrom_length=10000, gamma=1e-6, kernel_fun=kernel.get_kernel(), kernel_param=6, optimizer_max_iter=max_iterations, gpu=gpu)
    balance_constraint = (2 * 0.5) - 1  # 2r - 1
    tic = time()
    for idx, weight in enumerate([0.1, 0.15, 0.25, 1.]):
        print(" -> iteration {}".format(idx+1))
        lam0 = ((2/x_unlabeled.shape[0])*np.sum(functional_margin)) - (2*balance_constraint)
        y_u, lam, _iter = labelling(functional_margin, balance_constraint, lam0, 1., int(x_unlabeled.shape[0]*0.005))
        print("  -> [debug info] balance constraint {:.2}".format(np.divide(np.sum(y_u), x_unlabeled.shape[0])))
        print("  -> [debug info] lambda from {:.3e} to {:.3e} in {} iterations".format(lam0, lam, _iter+1))
        print("  -> [debug info] wrong labels {}".format(np.sum(y_u != y_unlabeled)))
        sample_weights = ([1.] * x_labeled.shape[0]) + ([weight] * x_unlabeled.shape[0])
        falkon.fit(np.vstack((x_labeled, x_unlabeled)), np.concatenate((y_labeled, y_u)).astype(np.float32), sample_weights=sample_weights)
        functional_margin = falkon.predict(x_unlabeled)
    print("Annealing done in {:.3} seconds".format(time()-tic))

    # testing falkon
    print("Starting falkon testing routine...")
    y_pred = falkon.predict(x_test)
    functional_margin = falkon.predict(x_test)
    accuracy = accuracy_score(y_test, np.sign(functional_margin))
    auc_roc = roc_auc_score(y_test, functional_margin)
    print("Accuracy: {:.3f} - AUC_ROC: {:.3f}".format(accuracy, auc_roc))    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", metavar='path', type=str, help='path of the dataset used for this test')
    parser.add_argument("--kernel", metavar='ker', type=str, default='gaussian', help='choose the kernel function')
    parser.add_argument("--max_iterations", type=int, default=20, help="specify the maximum number of iterations during the optimization")
    parser.add_argument("--gpu", type=bool, default=False, help='enable the GPU')

    args = parser.parse_args()

    main(path=args.dataset, kernel_function=args.kernel, max_iterations=args.max_iterations, gpu=args.gpu)
