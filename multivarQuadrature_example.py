# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 19:25:09 2024

@author: danie
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# Gauss-Hermite quadrature nodes and weights
x, w = np.polynomial.hermite.hermgauss(10)
print("Gauss-Hermite: %f" % np.sum(w))
print("Ground truth : %f" % np.sqrt(2 * np.pi))

# Multivariate normal parameters
mu = np.array([1, 0])
Sigma = np.array([[1.3, -0.213], [-0.213, 1.2]])
N = len(mu)

# Constant related to Gauss-Hermite normalization
const = np.pi**(-0.5*N)

# Multidimensional Gauss-Hermite quadrature setup
quadrature_nodes = np.array(list(itertools.product(*(x,) * N)))
quadrature_weights = np.prod(np.array(list(itertools.product(*(w,) * N))), axis=1)

# Transform the quadrature nodes to the multivariate normal distribution
transformed_nodes = np.sqrt(2.0) * np.dot(np.linalg.cholesky(Sigma), quadrature_nodes.T).T + mu[None, :]
print("Normalizing constant: %f" % np.sum(quadrature_weights * const))

# Mean calculation
mean = np.sum((quadrature_weights * const)[:, None] * transformed_nodes, axis=0)
print("Mean:")
print(mean)

# Covariance calculation
covfunc = lambda x: np.outer(x - mu, x - mu)
covariance = np.sum((quadrature_weights * const)[:, None, None] * np.array(list(map(covfunc, transformed_nodes))), axis=0)
print("Covariance:")
print(covariance)
