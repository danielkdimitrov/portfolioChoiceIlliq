"""
Created on Fri Oct  4 19:03:33 2024

@author: daniel


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from illiquidAssetModel import IlliquidAssetModel
from myPlots import *

# Define parameters
mu = np.array([0.067, 0.067, 0.067])  # Example: three assets
sigma = np.array([0.1626, 0.1626, 0.1626])
gamma = 6.0
beta = 0.031
r = 0.031
dt = .5

# Three-asset case with correlated assets
correlation_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.],
    [0.0, 0., 1.0]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix

# %%  Adjust if you run a variation with liquidity premia or with with Low Diversification

'Add Liquidity Premia'
mu[2] = mu[2] + +0.03

'Add correlation of the private asset'
correlation_matrix[2,1] = .8
correlation_matrix[1,2] = .8

# %% Run the model

# List of eta values to test
eta_values = [2, 1, 1/2, 1/5, 1/10, 1/15]

# Placeholder to store results
results = []

# Get Merton Results

# Run the model for each eta value
for eta in eta_values:
    print(f'1/eta={1/eta}')
    # Initialize and solve the model
    #dt = 1/2 if eta > 1 else 1    
    model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt, True)
    model.BellmanIterSolve()
    
    xi_sim_percentiles = np.percentile(model.xi_sim, [2.5, 50, 97.5])*100
    c_sim_percentiles = np.percentile(model.c_sim, [2.5, 50, 97.5])*100

    # Store the data in a DataFrame for each eta
    data = {
        '1/eta': 1/eta,  # Convert to percentage if necessary
        'Cash': model.alloc[0] * 100,  # Convert to percentage if necessary
        'Liquid 1': model.alloc[1] * 100,  # Convert to percentage if necessary
        'Liquid 2': model.alloc[2] * 100,  # Convert to percentage if necessary
        'Illiquid 1': model.alloc[3] * 100,  # Convert to percentage if necessary
        'Allocation Benefit': model.allocationBenefit_star*100,  # Repeat to match length
        'cec': model.cec_il_star*100,  # Repeat to match length
        'xi_sim_2.5%': xi_sim_percentiles[0],  # Repeat to match length
        'xi_sim_50%': xi_sim_percentiles[1] ,  # Repeat to match length
        'xi_sim_97.5%': xi_sim_percentiles[2],  # Repeat to match length
        'c_sim_2.5%': c_sim_percentiles[0],  # Repeat to match length
        'c_sim_50%': c_sim_percentiles[1],  # Repeat to match length
        'c_sim_97.5%': c_sim_percentiles[2],   # Repeat to match length
        'c': model.c_star_xi*100,   # Repeat to match length
        'convergence': model.convergence
    }
    df = pd.DataFrame(data, index =[0])

    # Append to the results list
    results.append(df)

# Merton 2m 
data = {
    '1/eta': '2m',  # Convert to percentage if necessary
    'Cash': model.alloc_2m[0] * 100,  # Convert to percentage if necessary
    'Liquid 1': model.alloc_2m[1] * 100,  # Convert to percentage if necessary
    'Liquid 2': model.alloc_2m[2] * 100,  # Convert to percentage if necessary    
    'cec': model.cec_m2*100,  # Repeat to match length
    'c': model.c_m2*100  # Repeat to match length
    }
df = pd.DataFrame(data, index =[0])
# Append to the results list
results.append(df)

# Merton 3m 
data = {
    '1/eta': '3m',  # Convert to percentage if necessary        
    'Cash': model.alloc_m[0] * 100,  # Convert to percentage if necessary
    'Liquid 1': model.alloc_m[1] * 100,  # Convert to percentage if necessary
    'Liquid 2': model.alloc_m[2] * 100,  # Convert to percentage if necessary    
    'Illiquid 1': model.alloc_m[3] * 100,  # Convert to percentage if necessary
    'cec': model.cec_m*100,  # Repeat to match length
    'c': model.c_m*100  # Repeat to match length
    }
df = pd.DataFrame(data, index =[0])
# Append to the results list
results.append(df)

# Combine all the DataFrames for easier comparison
final_df = pd.concat(results, axis=0)

# Save the final DataFrame as CSV for further analysis
final_df.to_csv("allocation_results_rho_80pc_gamma_6_lp03.csv", index=False)

print(final_df)

'''
TODO : 
    - Run with rho = 0
    - maybe run with lower gamma
    
    
'''

df = pd.read_excel('ModelData01.xlsx')

# Load the Excel file
df = pd.read_excel('ModelData01.xlsx')

def plot_allocation_benefit(lp_value, fileName):
    # Filter the data for lp=0 and ignoring 1/eta = 3m and 2m
    filtered_df = df[(df['lp'] == lp_value) & (df['1/eta'] != '3m') & (df['1/eta'] != '2m')]

    # Filter for rho=0 and rho=0.8
    rho_0 = filtered_df[filtered_df['rho'] == 0]
    rho_08 = filtered_df[filtered_df['rho'] == 0.8]

    # Get the Allocation Benefit for 3m in each case
    allocation_benefit_3m_rho_0 = df[(df['lp'] == lp_value) & (df['1/eta'] == '3m') & (df['rho'] == 0)]['Allocation Benefit'].values[0]
    allocation_benefit_3m_rho_08 = df[(df['lp'] == lp_value) & (df['1/eta'] == '3m') & (df['rho'] == 0.8)]['Allocation Benefit'].values[0]

    # Plotting
    plt.figure(figsize=(5, 4))
    plt.plot(rho_0['1/eta'], rho_0['Allocation Benefit'], 'purple', label='rho=0', marker ='o')
    plt.plot(rho_08['1/eta'], rho_08['Allocation Benefit'], 'blue',label='rho=0.8')

    # Adding dashed horizontal lines for Allocation Benefit for 3m
    plt.axhline(y=allocation_benefit_3m_rho_0, color='black', linestyle=':',alpha=.8, label='rho=0, 3m')
    plt.axhline(y=allocation_benefit_3m_rho_08, color='black', linestyle=':',alpha=.8,  label='rho=0.8, 3m')

    # Adding titles and labels
    plt.xlabel(r'$1/\eta$', fontsize=14)
    plt.ylabel('Allocation Benefit (%)',fontsize=14)
    plt.legend(['High Diversification','Low Diversification','Full Liquidity'],fontsize=12)
    plt.grid(True, alpha = .5)
    plt.tight_layout()
    saveFig(fileName)

# Run the function for lp=0
plot_allocation_benefit(0, 'allocationBenefitNoRP')

# Run the function for lp=0.03
plot_allocation_benefit(0.03, 'allocationBenefitRP')
