# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:35:05 2024

@author: NC4135
"""


import numpy as np
import pandas as pd

# Define parameters
mu = np.array([0.067, 0.067, 0.067])  # Example: three assets
sigma = np.array([0.1626, 0.1626, 0.1626])
gamma = 6.0
beta = 0.03
r = 0.02
dt = 1

# Three-asset case with correlated assets
correlation_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.8],
    [0.0, 0.8, 1.0]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix

# List of eta values to test
eta_values = [2, 1, 1/2, 1/5, 1/10, 1/15]

# Placeholder to store results
results = []

# Run the model for each eta value
for eta in eta_values:
    # Initialize and solve the model
    model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt, True)
    model.BellmanIterSolve()
    
    # Extract the allocation data after solving
    allocation = model.alloc  # Adjust this to match how `alloc` is accessed in your model

    # Labels for the allocation categories
    labels = ["Cash", "Liquid 1", "Liquid 2", "Illiquid"]

    # Store the data in a DataFrame for each eta
    data = {
        'Category': labels,
        f'Allocation (eta={eta})': allocation * 100  # Convert to percentage if necessary
    }
    df = pd.DataFrame(data)

    # Append to the results list
    results.append(df)

# Combine all the DataFrames for easier comparison
final_df = pd.concat(results, axis=1)

# Save the final DataFrame as CSV for further analysis
final_df.to_csv("allocation_results.csv", index=False)

print(final_df)