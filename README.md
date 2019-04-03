# An efficient implementation of the FALKON algorithm for Large Scale kernel methods and an extension to the semi-supervised scenario

Starting from one of the simplest kernel method (Kernel Ridge Regression) Rudi et al. have designed FALKON [1]. Falkon is one of the most efficient algorithm, from both computational and statistical points of view, able to work in a supervised large scale setting. This method is the result of a combination of three simple principles: sub-sampling, preconditioning and iterative solvers. Exploiting these ideas Falkon reaches sub-quadratic time complexity and linear memory requirements. The only weak spot in their work is represented by the high cost of data labelling, especially if concerned datasets are large. In order to overcome this problem we have designed an extension of Falkon able to work in a semi-supervised scenario (that is, a dataset made up of few labelled data and a lot of unlabelled ones) [2]. As we will see in the next chapters, our extension efficiently manages large semi-supervised datasets both from accuracy and time points of view.

## Semi-Supervised extension

## Usage

## Some results

## (Main) Reference papers
1. FALKON: An Optimal Large Scale Kernel Method - Alessandro Rudi, Luigi Carratino and Lorenzo Rosasco - [https://arxiv.org/abs/1705.10958](https://arxiv.org/abs/1705.10958)

2. Lagrangean-Based Combinatorial Optimization for Large Scale S3VMs - Francesco Bagattini, Paola Cappanera and Fabio Schoen - [https://ieeexplore.ieee.org/abstract/document/8113555](https://ieeexplore.ieee.org/abstract/document/8113555)
