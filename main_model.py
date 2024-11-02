# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:03:33 2024

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
from illiquidAssetModel import IlliquidAssetModel

from myPlots import plot_value_function, plot_allocation_chart


# %% Three asset case w/ correlated assets

# Define parameters
mu = np.array([0.055, 0.055, 0.055])  # Example: two liquid assets and one illiquid asset
sigma = np.array([0.14,0.14, 0.14])
correlation_matrix = np.array([
    [1.0,   0., 0. ],
    [0.,   1.0, 0.8],
    [0.,   0.8, 1. ]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix

gamma = 6.0
beta = 0.03
r = 0.02
dt = 1

'Run 10 Year model'
eta = 1/10

model_10year = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt, True)
model_10year.BellmanIterSolve()

print(f"10 year, xi_star: {model_10year.xi_star}")
model_10year.plot_results()

'Run 1 Year model'
eta = 1

model_1year = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model_1year.BellmanIterSolve()

print(f"1 year, xi_star: {model_1year.xi_star}")
model_1year.plot_results()

# Plot value fn comparison
plot_value_function(model_1year, model_10year)
# Plot stachs 
plot_allocation_chart(model_1year.alloc_m, model_1year.alloc, model_10year.alloc)




# %% Three asset case w/ uncorrelated assets

# Define parameters
mu = np.array([0.055, 0.055, 0.055])  # Example: two liquid assets and one illiquid asset
sigma = np.array([0.14,0.14, 0.14])
correlation_matrix = np.array([
    [1.0,   0., 0.],
    [0.,   1.0, 0.],
    [0.,   0., 1.]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix

gamma = 6.0
beta = 0.03
r = 0.02
dt = 1

'Run 10 Year model'
eta = 1/10

model_10year = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model_10year.BellmanIterSolve()

print(f"10 year, xi_star: {model_10year.xi_star}")
model_10year.plot_results()

'Run 1 Year model'
eta = 1

model_1year = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model_1year.BellmanIterSolve()

print(f"1 year, xi_star: {model_1year.xi_star}")
model_1year.plot_results()

# Plot value fn comparison
plot_value_function(model_1year, model_10year)
# Plot stachs 
plot_allocation_chart(model_1year.alloc_m, model_10year.alloc, model_10year.alloc)

# %% 
# Define parameters
mu = np.array([0.055, 0.055])  # Example: two liquid assets and one illiquid asset
sigma = np.array([0.14,0.14])
correlation_matrix = np.array([
    [1.0,   0.],
    [0.,   1.0]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix

gamma = 6.0
beta = 0.03
r = 0.02
dt = 1

'Run 10 Year model'
eta = 1/10

model_10year = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model_10year.BellmanIterSolve()

print(f"10 year, xi_star: {model_10year.xi_star}")
model_10year.plot_results()

'Run 1 Year model'
eta = 1

model_1year = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model_1year.BellmanIterSolve()

print(f"1 year, xi_star: {model_1year.xi_star}")
model_1year.plot_results()

plot_value_function(model_1year, model_10year)


# Assuming you already have the instances of the models created as model_1year and model_10year
# plot_value_function(model_1year, model_10year)
