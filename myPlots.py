# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:05:08 2024

@author: NC4135
"""

import numpy as np
import matplotlib.pyplot as plt



# Function to plot panel (a) Value Function
def plot_value_function(model1, model2):
    # Extract H values
    model1_values = -model1.ln_m_H_t_vals_opt_k  # H for 1 year
    model2_values = -model2.ln_m_H_t_vals_opt_k  # H for 10 years
    values_merton = -np.log(-model1.H_m)  # H continuous trading (use 1year as representative)
    
    # Create the ξ grid
    xi_grid = model1.Xi_t

    # Diamond values
    xi_diamond_1Y = model1.xi_star
    H_diamond_1Y = -model1.ln_m_H_star

    xi_diamond_10Y = model2.xi_star
    H_diamond_10Y = -model2.ln_m_H_star

    # Plot
    plt.figure(figsize=(5, 4))
    
    # Plot 1 year (solid line)
    plt.plot(xi_grid, model1_values, 'b-', label='1 Year Friction')
    
    # Plot 10 year (dotted line)
    plt.plot(xi_grid, model2_values, 'b:', label='10 Year Friction')
    
    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton, color='gray', linestyle='--', label='Continuous Trading')
    
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
    
# Function to plot panel CEC gain
def cec_gain(model1, model2):
    # Extract H values
    model1_values = 100* model1.allocationBenefit  # H for 1 year
    model2_values = 100* model2.allocationBenefit # H for 10 years
    values_merton = 100* (model1.getCec(model1.H_m)/ model1.cec_m2 -1)
    
    # Create the ξ grid
    xi_grid = model1.Xi_t

    # Diamond values
    xi_diamond_1 = model1.xi_star
    H_diamond_1 = 100 * model1.allocationBenefit_star
    xi_diamond_2 = model2.xi_star
    H_diamond_2 = 100* model2.allocationBenefit_star

    # Plot
    plt.figure(figsize=(5, 4))
    
    # Plot 1 year (solid line)
    plt.plot(xi_grid, model1_values, 'b-', label='1 Year Friction')
    
    # Plot 10 year (dotted line)
    plt.plot(xi_grid, model2_values, 'purple',':', label='10 Year Friction')
    
    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton, color='gray', linestyle='--', label='Continuous Trading')
    plt.axhline(0, color='gray', linestyle='-' )
    
    # Plot diamond point
    plt.plot(xi_diamond_1, H_diamond_1, 'D', color='b')
    plt.plot(xi_diamond_2, H_diamond_2, 'D', color='purple')

    # Formatting the plot
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$CEC Gain (\%)$')
    plt.title('Investment Gain')
    plt.legend()
    
    plt.ylim(bottom=-50)  # Set the lower bound on the y-axis
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()    
    
def plot_value_function_1m(model1):
    # Create the ξ grid
    xi_grid = model1.xi_fine_grid

    # Extract H values
    model1_values = -model1.ln_m_H_func(xi_grid)  # H for 1 year
    values_merton = -np.log(-model1.H_m)  # H continuous trading (use 1 year as representative)
    values_merton_H2 = -np.log(-model1.H_m2)  # H continuous trading (use 1 year as representative)

    # Diamond values
    xi_diamond_1Y = model1.xi_star
    H_diamond_1Y = -model1.ln_m_H_star
    
    # Calculate the 95% range for model1.xi_sim
    xi_sim_95_low, xi_sim_50, xi_sim_95_high = np.percentile(model1.xi_sim, [2.5, 50, 97.5])

    # Median values
    H_median = -model1.ln_m_H_func(xi_sim_50)
    
    # Plot
    plt.figure(figsize=(5, 4))
    
    # Plot 1 year (solid line)
    plt.plot(xi_grid, model1_values, 'b-', label='1 Year Friction')
    
    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton, color='gray', linestyle='--', label='Continuous Trading with Private Asset')
    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton_H2, color='gray', linestyle=':', label='Continuous Trading ex. Private Asset')

    # Plot diamond point
    plt.plot(xi_diamond_1Y, H_diamond_1Y, 'D', color='b')
    # Plot diamond point
    plt.plot(xi_sim_50, H_median, '|', color='black', markersize=15, markeredgewidth=2)

    # Grey out the 95% range over the x-axis
    plt.fill_between(xi_grid, values_merton, -25, where=((xi_grid >= xi_sim_95_low) & (xi_grid <= xi_sim_95_high)), color='gray', alpha=0.3, label='95% Range')

    # Formatting the plot
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$-\log(-H(\xi))$')
    plt.title('Value Function')
    plt.legend()
    
    plt.ylim(top=-17)  # Set the lower bound on the y-axis
    plt.ylim(bottom=-25)  # Set the lower bound on the y-axis
    #plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_theta_function_1m(model1):
    # Create the ξ grid
    xi_grid = model1.xi_fine_grid

    # Extract H values
    model1_values1 = model1.theta_func[0](xi_grid)*(1-xi_grid)  # H for 1 year
    model1_values2 = model1.theta_func[1](xi_grid)*(1-xi_grid)  # H for 1 year    
    values_merton1 = model1.pi_m[0]  # H continuous trading (use 1 year as representative)
    values_merton2 = model1.pi_m[1]  # H continuous trading (use 1 year as representative)

    #values_merton_H2 = -np.log(-model1.H_m2)  # H continuous trading (use 1 year as representative)

    # Diamond values
    xi_diamond_1Y = model1.xi_star
    theta_star_xi1 = model1.theta_star_xi[0]*(1-xi_diamond_1Y)
    theta_star_xi2 = model1.theta_star_xi[1]*(1-xi_diamond_1Y)

    # Calculate the 95% range for model1.xi_sim
    xi_sim_95_low, xi_sim_50, xi_sim_95_high = np.percentile(model1.xi_sim, [2.5, 50, 97.5])

    # Median values
    #theta_median = -model1.theta_func[0](xi_sim_50)
    
    # Plot
    plt.figure(figsize=(5, 4))
    
    # Plot 1 year (solid line)
    plt.plot(xi_grid, model1_values1, 'b-', label='Uncorrelated Asset')
    plt.plot(xi_grid, model1_values2, 'grey',':', label='Correlated Asset')

    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton1, color='gray', linestyle='--', label='Continuous Trading')
    plt.axhline(values_merton2, color='gray', linestyle='--', label='Continuous Trading')

    # Plot continuous (dashed line, horizontal)
    #plt.axhline(values_merton_H2, color='gray', linestyle=':', label='Liquid Assets')

    # Plot diamond point
    plt.plot(xi_diamond_1Y, theta_star_xi1, 'D', color='b')
    plt.plot(xi_diamond_1Y, theta_star_xi2, 'D', color='grey')
    # Plot diamond point
    #plt.plot(xi_sim_50, H_median, '|', color='black', markersize=15, markeredgewidth=2)

    # Grey out the 95% range over the x-axis
    plt.fill_between(xi_grid, values_merton1, 0, where=((xi_grid >= xi_sim_95_low) & (xi_grid <= xi_sim_95_high)), color='gray', alpha=0.3, label='95% Range')

    # Formatting the plot
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$-\log(-H(\xi))$')
    plt.title('Liquid Risk Allocation')
    plt.legend()
    
    #plt.ylim(top=-17)  # Set the lower bound on the y-axis
    plt.ylim(bottom=0.01)  # Set the lower bound on the y-axis
    #plt.grid(True)
    plt.tight_layout()
    plt.show()    

def plot_alloc_tact(model1, model2, j):
    # Extract H values
    model1_values = model1.theta_opt[:,j]*(1-model1.Xi_t)  # H for 1 year
    model2_values = model2.theta_opt[:,j]*(1-model2.Xi_t)  # H for 10 years
    values_merton = model1.pi_m[j]  # H continuous trading (use 1year as representative)
    
    # Create the ξ grid
    xi_grid = model1.Xi_t

    # Diamond values
    xi_diamond_1Y = model1.xi_star
    H_diamond_1Y = -model1.theta_star_xi*(1-xi_diamond_1Y)

    xi_diamond_10Y = model2.xi_star
    H_diamond_10Y = -model2.theta_star_xi*(1-xi_diamond_10Y)

    # Plot
    plt.figure(figsize=(5, 4))
    
    # Plot 1 year (solid line)
    plt.plot(xi_grid, model1_values, 'b-', label='Model 1')
    
    # Plot 10 year (dotted line)
    plt.plot(xi_grid, model2_values, 'b:', label='Model 2')
    
    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton, color='gray', linestyle='--', label='Continuous Trading')
    
    # Plot diamond point
    plt.plot(xi_diamond_1Y, H_diamond_1Y, 'D', color='b')
    plt.plot(xi_diamond_10Y, H_diamond_10Y, 'D', color='purple')

    # Formatting the plot
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$\theta(1-\xi)$')
    plt.title('Liquid Allocation')
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
