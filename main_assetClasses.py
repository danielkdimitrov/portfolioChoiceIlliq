# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:59:49 2024

@author: NC4135
"""

import numpy as np
import matplotlib.pyplot as plt
from illiquidAssetModel import IlliquidAssetModel

from myPlots import *
import seaborn as sns
import pandas as pd

# %% # Function to make a matrix positive semidefinite

def nearest_positive_semidefinite(matrix, epsilon=1e-10):
    """Find the nearest positive semidefinite matrix to the input matrix using the Higham algorithm."""
    # Compute the symmetric part of the matrix
    sym_matrix = (matrix + matrix.T) / 2

    # Perform eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(sym_matrix)

    # Set any negative eigenvalues to a small positive value
    eigvals[eigvals < epsilon] = epsilon

    # Reconstruct the matrix with non-negative eigenvalues
    psd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Ensure the matrix is symmetric
    psd_matrix = (psd_matrix + psd_matrix.T) / 2

    return psd_matrix

def collect_output(selected_assets, models,fileName):
    # Asset classes including Money Market
    asset_classes = ['Money Market'] + selected_assets
    
    model_liq, model_illiq = models
    # Ensure the lengths match
    # Create the dataframe
    df = pd.DataFrame({
        'Asset Class': asset_classes,
        'Allocation Liq': model_liq.alloc*100, 
        'Allocation Illiq': model_illiq.alloc*100
    })
    
    # Add the new row
    df.loc[len(df)] = ['cec', model_liq.cec_il_star * 100, model_illiq.cec_il_star * 100]

    df.loc[len(df)] = ['xi_med', np.percentile(model_liq.xi_sim, 50) * 100, np.percentile(model_illiq.xi_sim, 50) * 100]
    df.loc[len(df)] = ['c_med', np.percentile(model_liq.c_sim, 50) * 100, np.percentile(model_illiq.c_sim, 50) * 100]
    # Add allocation benefit relative to xi = 0
    df.loc[len(df)] = ['allocation benefit',(model_liq.cec_il_star/ model_illiq.cec_il[0] -1)*100 , (model_illiq.cec_il_star/ model_illiq.cec_il[0] -1)*100]
    
    # Interpolate the cec_il values
    interpolator = interp1d(model_illiq.Xi_t, model_illiq.cec_il)
    
    # Find the hypothetical value of cec_il when Xi_t is equal to xi_star
    hypothetical_cec_il = interpolator(model_liq.xi_star)

   # Add missallocation loss relative to xi = xi(full liq.)
    df.loc[len(df)] = ['missalocation cost', 0 , (hypothetical_cec_il / model_illiq.cec_il_star -1)*100]
   
    print(df)
            
    # Save the final DataFrame as CSV for further analysis
    df.to_csv(fileName+".csv", index=False)

def setIlliquidAssetList(currentAC):
    # Load the CSV file
    file_path = r'data\inputData.csv'
    df = pd.read_csv(file_path)
    
    # Extract the relevant rows and columns
    selected_assets = [
        'U.S. Long Treasuries',
        'U.S. Long Corporate Bonds',
        #'U.S. Aggregate Bonds', 
        #'World ex-U.S. Government Bonds hedged',
        #'U.S. High Yield Bonds', 
        'U.S. Large Cap', 
        'U.S. Small Cap',
        'EAFE Equity',
        'U.S. REITs',
        'U.S. Core Real Estate',
        'Private Equity',
        'Global Core Infrastructure',
        'Diversified Hedge Funds',
        'Macro Hedge Funds'
    ]
    # Case switch logic
    if currentAC == 'U.S. Core Real Estate':
        assets_to_drop = ['Private Equity',
                'Global Core Infrastructure',
                'Diversified Hedge Funds',
                'Macro Hedge Funds']
    elif currentAC == 'Private Equity':
        assets_to_drop = ['U.S. Core Real Estate',
        'Global Core Infrastructure',
        'Diversified Hedge Funds',
        'Macro Hedge Funds']
    elif currentAC == 'Global Core Infrastructure':
        assets_to_drop = ['U.S. Core Real Estate',
        'Private Equity',
        'Diversified Hedge Funds',
        'Macro Hedge Funds']
    elif currentAC == 'Diversified Hedge Funds':
        assets_to_drop = ['U.S. Core Real Estate',
        'Private Equity',
        'Global Core Infrastructure',
        'Macro Hedge Funds']
    elif currentAC == 'Macro Hedge Funds':
        assets_to_drop = ['U.S. Core Real Estate',
        'Private Equity',
        'Global Core Infrastructure',
        'Diversified Hedge Funds']  
    else:
        assets_to_drop = []
    
    # Drop the specified assets
    selected_assets = [asset for asset in selected_assets if asset not in assets_to_drop]
    
    df_selected = df[df['Asset Class'].isin(selected_assets)]
    
    # Extract 'Compound Return (%)' as numpy array mu
    mu = df_selected['Compound Return (%)'].values/100
    
    # Extract 'Annualized Volatility (%)' as numpy array sigma
    sigma = df_selected['Annualized Volatility (%)'].values/100
    
    # Extract the correlation matrix for the selected assets
    correlation_matrix = df_selected[selected_assets].values
    
    # Set the diagonal elements to ones and mirror the lower diagonal elements
    for i in range(len(correlation_matrix)):
        for j in range(i, len(correlation_matrix)):
            if i == j:
                correlation_matrix[i][j] = 1.0
            else:
                correlation_matrix[i][j] = correlation_matrix[j][i]
    
    Sigma = np.outer(sigma, sigma) * correlation_matrix
    
    # Make the correlation matrix positive semidefinite
    Sigma_psd = nearest_positive_semidefinite(Sigma)
    
    print('Eigenvalues of the adjusted cov matrix: ', np.linalg.eigvals(Sigma_psd))
    print('L1 Norm of the adjusted vs non-adjusted: ',np.linalg.norm(Sigma - Sigma_psd, 'fro')**2)
    return selected_assets, mu, Sigma_psd, correlation_matrix


# %% PARAMETERS 
gamma = 6.0
beta = 0.031
r = 0.031
dt = 1.

'''
# Define parameters
mu = np.array([4.6, 6.7, 9.9])/100
sigma = np.array([4.52, 16.26, 19.62])/100

correlation_matrix = np.array([
    [1.0, 0.26, 0.0],
    [0.26, 1.0, 0.78],
    [0.0, 0.78, 1.0]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix

# %% Three asset case w/ correlated assets


'Run 10 Year model'

model_10year = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt, True)
model_10year.BellmanIterSolve()
print(f"10 year, xi_star: {model_10year_corr.xi_star}")
model_10year.plot_results()

'''


# %% Illiquid Case

asset_class = 'Private Equity'
selected_assets, mu, Sigma_psd, correlation_matrix = setIlliquidAssetList(asset_class)

eta =1/5 #1/10

model_10year_full = IlliquidAssetModel(mu, Sigma_psd, gamma, beta, eta, r, dt, True, True)
print('Unconstrained Merton Allocations:', model_10year_full.alloc_m)
model_10year_full.BellmanIterSolve()
print(f"10 year, xi_star: {model_10year_full.xi_star}")
model_10year_full.plot_results()
print(f"10 year, SAA: {model_10year_full.alloc}")


# %% Liquid Case

selected_assets, mu, Sigma_psd, correlation_matrix = setIlliquidAssetList(asset_class)

eta = 5
model_10year_liq = IlliquidAssetModel(mu, Sigma_psd, gamma, beta, eta, r, dt, True, True)
print('Unconstrained Merton Allocations:', model_10year_liq.alloc_m)
model_10year_liq.BellmanIterSolve()
print(f"xi_star: {model_10year_liq.xi_star}")
model_10year_liq.plot_results()
print(f"SAA: {model_10year_liq.alloc}")

# %%
collect_output(selected_assets, [model_10year_liq, model_10year_full],asset_class)
# %%
