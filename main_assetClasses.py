# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:59:49 2024

@author: NC4135
"""

import numpy as np
import matplotlib.pyplot as plt
from illiquidAssetModel import IlliquidAssetModel, plot_cec

from myPlots import plot_cec


##########

'Baseline Params'
r = 1.3/100
gamma = 6.0
beta = 0.03

'Do this hedge funds'

# Updated vector of expected returns
eta = 1.0

mu = np.array([2.44, 9.66, 3.82])/100  # GB, HF, Pr

# Updated covariance matrix
sigma = np.array([14.00, 18.68, 6.84])/100  # GB, HF, Pr
correlation_matrix = np.array([
    [1.0,   -0.3,  -0.33],
    [-0.3,  1.0,    0.76],
    [-0.33,  0.76,   1.0]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix


model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r)
model.BellmanIterSolve()
plot_cec(model)

'Merton 1 risky assets PuE'
mu = np.array([6.17])/100


sigma = np.array([ 12.75])/100
correlation_matrix = np.array([
    [1.0 ]
])

Sigma = np.outer(sigma, sigma) * correlation_matrix
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r)

'Merton 1 risky assets GB'
mu = np.array([2.44])/100


sigma = np.array([14.00])/100
correlation_matrix = np.array([
    [1.0 ]
])

Sigma = np.outer(sigma, sigma) * correlation_matrix
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r)


'Merton 2 risky assets'

mu = np.array([2.44, 6.17])/100


sigma = np.array([14.00, 12.75])/100
correlation_matrix = np.array([
    [1.0,   -0.3],
    [-0.3,   1.0]
])

Sigma = np.outer(sigma, sigma) * correlation_matrix
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r)

'Three asset case with Private Equity'
mu = np.array([2.44, 6.17, 9.66])/100


sigma = np.array([14.00, 12.75, 18.68])/100
correlation_matrix = np.array([
    [1.0,   -0.3,  -0.57],
    [-0.3,   1.0,   0.84],
    [-0.57,  0.84,  1.0]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix


# Initialize the model with the parameters
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r)

# Solve the model
model.BellmanIterSolve()

# Access results, for example, H_t_vals_opt_k or consumption paths
print(model.xi_star)
plot_cec(model)



