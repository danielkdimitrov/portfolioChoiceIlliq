import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from illiquidAssetModel import IlliquidAssetModel
from myPlots import *

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
eta_values = [2, 1, 1/2, 1/5, 1/10, 1/15, 1/20]

# Placeholder to store results
results = []

# Get Merton Results

# Run the model for each eta value
for eta in eta_values:
    print(f'1/eta={1/eta}')
    # Initialize and solve the model
    dt = 1/2 if eta > 1 else 1    
    model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt, True)
    model.BellmanIterSolve()
    
    xi_sim_percentiles = np.percentile(model.xi_sim, [2.5, 50, 97.5])
    c_sim_percentiles = np.percentile(model.c_sim, [2.5, 50, 97.5])

    # Store the data in a DataFrame for each eta
    data = {
        '1/eta': 1/eta,  # Convert to percentage if necessary
        'Cash': model.alloc[0] * 100,  # Convert to percentage if necessary
        'Liquid 1': model.alloc[1] * 100,  # Convert to percentage if necessary
        'Liquid 2': model.alloc[2] * 100,  # Convert to percentage if necessary
        'Illiquid 1': model.alloc[3] * 100,  # Convert to percentage if necessary
        'Allocation Benefit': [model.allocationBenefit_star],  # Repeat to match length
        'self.cec_il_star': [model.cec_il_star],  # Repeat to match length
        'xi_sim_2.5%': [xi_sim_percentiles[0]],  # Repeat to match length
        'xi_sim_50%': [xi_sim_percentiles[1]] ,  # Repeat to match length
        'xi_sim_97.5%': [xi_sim_percentiles[2]],  # Repeat to match length
        'c_sim_2.5%': [c_sim_percentiles[0]],  # Repeat to match length
        'c_sim_50%': [c_sim_percentiles[1]],  # Repeat to match length
        'c_sim_97.5%': [c_sim_percentiles[2]]   # Repeat to match length
    }
    df = pd.DataFrame(data)

    # Append to the results list
    results.append(df)

# Combine all the DataFrames for easier comparison
final_df = pd.concat(results, axis=0)

# Save the final DataFrame as CSV for further analysis
final_df.to_csv("allocation_results.csv", index=False)

print(final_df)

'''
TODO : 
    - add the merton solution in there after the loop (2 asset and 3 asset )
    - check why alloc benefit negative but still allocated to xi - 1/eta = 20 
        - make tolerance smaller? 
    - Run with rho = 0
    - maybe run with lower gamma
    
    
'''

