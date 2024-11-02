# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:05:08 2024

@author: NC4135
"""

import numpy as np
import matplotlib.pyplot as plt



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
    
    plt.ylim(bottom=-25)  # Set the lower bound on the y-axis
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_allocation_chart(model_alloc_m, model_alloc1, model_alloc2):
    # Convert to percentage
    model_alloc_m = np.array(model_alloc_m) * 100
    model_alloc1 = np.array(model_alloc1) * 100
    model_alloc2 = np.array(model_alloc2) * 100

    # Labels, colors, and hatching patterns for readability in black and white
    labels = ["cash", "liq 1", "liq 2", "illiq"]
    colors = ['#4e79a7', '#a0cfa2', '#e15759', '#f1ce63']
    hatches = ['', '////', '....', 'xxxx']  # Different hatch patterns for each category

    # Set X-axis positions for the three bars, with narrower spacing
    x = np.array([0, 1, 2]) * 0.6  # Reduce spacing by reducing the multiplier here

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each stacked bar for model_alloc_m, model_alloc1, and model_alloc2
    for idx, (model, x_pos) in enumerate(zip([model_alloc_m, model_alloc1, model_alloc2], x)):
        bottom = 0
        for i in range(len(model)):
            ax.bar(x_pos, model[i], bottom=bottom, color=colors[i], edgecolor='white', 
                   width=0.3, hatch=hatches[i], label=labels[i] if idx == 0 else "")
            bottom += model[i]

            # Add text labels in the middle of each section
            ax.text(x_pos, bottom - model[i] / 2, f"{model[i]:.1f}%", ha='center', color='black', fontsize=12)

    # Labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(['Merton Model', 'Model with Illiquidity', 'Alternative Model'], fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Allocation Percentage', fontsize=14)
    ax.legend(labels, loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=12)

    # Increase general font size for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Display the plot
    plt.tight_layout()
    plt.show()
    


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
