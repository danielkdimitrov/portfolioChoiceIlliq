# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:03:33 2024

@author: danie
"""
import numpy as np
import matplotlib.pyplot as plt
from illiquidAssetModel import IlliquidAssetModel

# Function to plot panel (a) Value Function
def plot_value_function(model_1year, model_10year):
    # Extract H values
    H_1year = -model_1year.ln_m_H_t_vals_opt_k  # H for 1 year
    H_10year = -model_10year.ln_m_H_t_vals_opt_k  # H for 10 years
    H_cont = -np.log(-model_1year.H_m)  # H continuous trading (use 1year as representative)
    
    # Create the ξ grid
    xi_grid = model_1year.Xi_t

    # Diamond values
    xi_diamond_1Y = model_1year.xi_star
    H_diamond_1Y = -model_1year.ln_m_H_star

    xi_diamond_10Y = model_10year.xi_star
    H_diamond_10Y = -model_10year.ln_m_H_star

    # Plot
    plt.figure(figsize=(5, 4))
    
    # Plot 1 year (solid line)
    plt.plot(xi_grid, H_1year, 'b-', label='1 Year Friction')
    
    # Plot 10 year (dotted line)
    plt.plot(xi_grid, H_10year, 'b:', label='10 Year Friction')
    
    # Plot continuous (dashed line, horizontal)
    plt.axhline(H_cont, color='gray', linestyle='--', label='Continuous Trading')
    
    # Plot diamond point
    plt.plot(xi_diamond_1Y, H_diamond_1Y, 'D', color='b')
    plt.plot(xi_diamond_10Y, H_diamond_10Y, 'D', color='purple')

    # Formatting the plot
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$-\log(-H(\xi))$')
    plt.title('Value Function')
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% Three asset case

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

plot_value_function(model_1year, model_10year)










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
