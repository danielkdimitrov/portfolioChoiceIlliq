# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:05:08 2024

@author: NC4135
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d


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
    
def plot_value_function_1m(model1, limitUp, limitDown, save=False, fileName='chart'):
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
    #plt.title('Value Function')
    plt.legend()
    
    plt.ylim(top=limitUp)  # Set the lower bound on the y-axis
    plt.ylim(bottom=limitDown)  # Set the lower bound on the y-axis
    #plt.grid(True)
    plt.tight_layout()
    if save:
        saveFig(fileName)
    else: 
        plt.show()

def plot_c(model1, limitUp, limitDown, save=False, fileName='chart', legend=True, smooth=False):
    # Create the ξ grid
    xi_grid = model1.xi_fine_grid

    # Extract H values
    model1_values = model1.c_func(model1.xi_fine_grid)*(1-model1.xi_fine_grid)*100  # H for 1 year
    values_merton = model1.c_m*100  # H continuous trading (use 1 year as representative)
    values_merton_H2 = model1.c_m2*100  # H continuous trading (use 1 year as representative)
    
    if smooth == True:
        'Interpolate data at the edges if needed'
        # Remove values smaller than 2 but keep the last value
        filtered_data = np.where(model1_values[:-1] < 2, np.nan, model1_values[:-1])
        filtered_data = np.append(filtered_data, model1_values[-1])
        
        # Get indices of valid (non-NaN) values
        valid_indices = np.where(~np.isnan(filtered_data))[0]
        
        # Interpolate to fill NaN values
        interpolator = interp1d(valid_indices, filtered_data[valid_indices], kind='linear', fill_value="extrapolate")
        model1_values = interpolator(np.arange(len(filtered_data)))

    # Diamond values
    xi_diamond_1Y = model1.xi_star
    H_diamond_1Y = model1.c_star_xi*100
    
    # Calculate the 95% range for model1.xi_sim
    xi_sim_95_low, xi_sim_50, xi_sim_95_high = np.percentile(model1.xi_sim, [2.5, 50, 97.5])

    # Median values
    H_median = np.percentile(model1.c_sim, 50)*100
    
    # Plot
    plt.figure(figsize=(4, 3))
    
    # Plot 1 year (solid line)
    plt.plot(xi_grid, model1_values, 'b-', label='Illiquid Private Asset')
    
    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton, color='gray', linestyle='--', label='Continuous Trading with Private Asset')
    # Plot continuous (dashed line, horizontal)
    plt.axhline(values_merton_H2, color='gray', linestyle=':', label='Continuous Trading ex. Private Asset')

    # Plot diamond point
    plt.plot(xi_diamond_1Y, H_diamond_1Y, 'D', color='b')
    # Plot diamond point
    plt.plot(xi_sim_50, H_median, '|', color='black', markersize=15, markeredgewidth=2)

    # Grey out the 95% range over the x-axis
    plt.fill_between(xi_grid, values_merton, limitDown, where=((xi_grid >= xi_sim_95_low) & (xi_grid <= xi_sim_95_high)), color='gray', alpha=0.3, label='95% Range')

    # Formatting the plot
    plt.xlabel(r'$\xi$')
    plt.ylabel('Consumption Rate (%)')
    #plt.title('Value Function')
    if legend == True: 
        plt.legend()
    
    plt.ylim(top=limitUp)  # Set the lower bound on the y-axis
    plt.ylim(bottom=limitDown)  # Set the lower bound on the y-axis
    #plt.grid(True)
    plt.tight_layout()
    if save:
        saveFig(fileName)
    else: 
        plt.show()

        
def plot_theta_function_1m(model1, save=False, fileName='chart'):
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
    if save:
        saveFig(fileName)
    else: 
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
    

def plot_allocation_chart(model_alloc1_m, model_alloc1, model_alloc2_m, model_alloc2, save=False, fileName='allocation_chart'):
    # Convert to percentage
    model_alloc1_m = np.array(model_alloc1_m) * 100
    model_alloc1 = np.array(model_alloc1) * 100
    model_alloc2_m = np.array(model_alloc2_m) * 100
    model_alloc2 = np.array(model_alloc2) * 100

    # Labels, colors, and hatching patterns for readability in black and white
    labels = ["Cash", "Liquid 1", "Liquid 2", "Illiquid"]
    colors = ['#4e79a7', '#a0cfa2', '#f1ce63', '#e15759']
    hatches = ['', '\\', '//', '|']  # Different hatch patterns for each category

    # Set X-axis positions for the three bars, with narrower spacing
    x = np.array([0, 1, 2, 3]) * 0.6  # Reduce spacing by reducing the multiplier here

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each stacked bar for model_alloc_m, model_alloc1, and model_alloc2
    for idx, (model, x_pos) in enumerate(zip([model_alloc1_m, model_alloc1, model_alloc2_m, model_alloc2], x)):
        bottom = 0
        for i in range(len(model)):
            ax.bar(x_pos, model[i], bottom=bottom, color=colors[i], edgecolor='white', 
                   width=0.3, hatch=hatches[i], label=labels[i] if idx == 0 else "")
            bottom += model[i]

            # Add text labels in the middle of each section
            ax.text(x_pos, bottom - model[i] / 2, f"{model[i]:.1f}%", ha='center', color='black', fontsize=12)

    # Labels and legend
    ax.set_xticks(x)
    ax.set_xticklabels(['Continuous', 'Illiquid', 'Continuous Corr.', 'Illiquid Corr.'], fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Allocation Percentage', fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=12)

    # Increase general font size for better readability
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Display the plot
    plt.tight_layout()
    if save:
        # Collect data into a DataFrame
        data = {
        'Category': labels,
        'Model Alloc 1 M': model_alloc1_m,
        'Model Alloc 1': model_alloc1,
        'Model Alloc 2 M': model_alloc2_m,
        'Model Alloc 2': model_alloc2
        }
        df = pd.DataFrame(data)
        saveFig(fileName, df)

    else: 
        plt.show()

    
def plot_cec(model,legend = False, save=False, fileName='allocation_chart'):
    """
    Plots the Certainty Equivalent Consumption (CEC) as a function of xi,
    with a dashed vertical line at xi_star.
    """
    # Example CEC curve (replace with actual computations)
    H_t_vals_opt_k = -np.exp(model.ln_m_H_func(model.xi_fine_grid))  # This is just a placeholder
    
    cec_vals = model.getCec(H_t_vals_opt_k)
    
    plt.figure(figsize=(4, 3))
    # Create the plot
    plt.plot(model.xi_fine_grid, cec_vals*100, color='blue', label =r'$CEC(\xi)$')

    # Add the certical dashed line at cec_m2
    plt.axvline(x=0, color='gray', linestyle=':',label=r'$\xi=0$')
    
    # Add the vertical dashed line at xi_star
    plt.axvline(x=model.xi_star, color='gray', alpha=0.8, linestyle='-', label=r'$\xi^*$')

    # Add the vertical dashed line at pi_m[-1]
    plt.axvline(x=model.pi_m[-1], color='gray', alpha=0.99, linestyle='--',label=r'$\xi^{Liquid}$')
            
    # Labeling
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$CEC(\xi) (\%)$')
    if legend == True:
        plt.legend()
    if save:
        saveFig(fileName)
    else:
        plt.title(r'Certainty Equivalent Consumption (CEC) vs. $\xi$')
        plt.show()

def saveFig(fileName, df=None):
    # Get the current folder path
    current_folder = os.getcwd()
    
    # Add the subfolder 'images' to the current folder path
    images_folder = os.path.join(current_folder, 'images')
    
    # Ensure the 'images' folder exists
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    # Construct the full file path and save the figure in different formats
    for ext in ['.pdf', '.png', '.JPG', '.TIF']:
        file_path = os.path.join(images_folder, fileName + ext)
        plt.savefig(file_path, bbox_inches='tight')
    if df is not None:
        # Save the DataFrame to an Excel file
        excel_file_path = os.path.join(images_folder, fileName + '.xlsx')
        df.to_excel(excel_file_path, index=False)

# Example usage
# saveFig('example_plot', df)

     
