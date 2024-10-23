# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:59:49 2024

@author: NC4135
"""

import numpy as np
import matplotlib.pyplot as plt
from illiquidAssetModel import IlliquidAssetModel



def plot_cec(model):
     """
     Plots the Certainty Equivalent Consumption (CEC) as a function of xi,
     with a dashed vertical line at xi_star.
     """
     # Example CEC curve (replace with actual computations)
     H_t_vals_opt_k = -np.exp( model.ln_m_H_func(model.xi_fine_grid))  # This is just a placeholder
     
     cec_vals = model.getCec(H_t_vals_opt_k)
     
     # Create the plot
     plt.plot(model.xi_fine_grid, cec_vals, color='blue')
     
     # Add the vertical dashed line at xi_star
     plt.axvline(x=model.xi_star, color='gray', linestyle='--')
     
     # Labeling
     plt.xlabel(r'$\xi$')
     plt.ylabel(r'$CEC(\xi)$')
     plt.show()

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
model.solve()

# Access results, for example, H_t_vals_opt_k or consumption paths
print(model.xi_star)
plot_cec(model)



