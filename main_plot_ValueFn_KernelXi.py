import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from illiquidAssetModel import IlliquidAssetModel
from myPlots import *
import seaborn as sns


# %%
# Define parameters
mu = np.array([0.067, 0.067, 0.067])  # Example: three assets
sigma = np.array([0.1626, 0.1626, 0.1626])
gamma = 6.0
beta = 0.031
r = 0.031
dt = 1.

'If liquidity premia is added'
#mu[2] = mu[2]+.03

# Three-asset case with correlated assets
correlation_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.],
    [0.0, 0., 1.0]
])
Sigma = np.outer(sigma, sigma) * correlation_matrix

# List of eta values to test
one_over_eta_grid = np.linspace(1, 15, 15) #[5, 15] #

# %%
# Placeholder to store results
ln_m_H_t_vals_opt_k_results = []
theta_opt_results = []
c_opt_results = []
c_sim_results = []
xi_sim_results = []
xi_star_results = []
ln_m_H_star_results = []

# Run the model for each eta value
for one_over_eta in one_over_eta_grid:
    print(f'1/eta={one_over_eta}')
    # Initialize and solve the model
    model = IlliquidAssetModel(mu, Sigma, gamma, beta, 1/one_over_eta, r, dt, True)
    model.BellmanIterSolve()
    
    # Collect the values
    ln_m_H_t_vals_opt_k_results.append(model.ln_m_H_t_vals_opt_k)
    theta_opt_results.append(model.theta_opt[:, 0]*(1-model.Xi_t))
    c_opt_results.append(model.c_opt*(1-model.Xi_t))
    c_sim_results.append(model.c_sim)
    xi_sim_results.append(model.xi_sim)
    xi_star_results.append(model.xi_star)
    ln_m_H_star_results.append(model.ln_m_H_star)

# Convert lists to numpy arrays for easier manipulation
ln_m_H_t_vals_opt_k_results = np.array(ln_m_H_t_vals_opt_k_results)
xi_star_results = np.array(xi_star_results)

c_vals_opt_k_results = np.array(c_opt_results)
theta_vals_results = np.array(theta_opt_results)



# %%

# Create a grid for eta values and xi_sim values
one_over_eta_mesh_grid, xi_mesh_grid = np.meshgrid(one_over_eta_grid, model.Xi_t)


# Create a 3D plot
fig = plt.figure(figsize=(8,12))
ax = fig.add_subplot(111, projection='3d')

# Plot the data using plot_wireframe
#ax.plot_wireframe(one_over_eta_mesh_grid, xi_mesh_grid*100, -ln_m_H_t_vals_opt_k_results.T)

#ax.plot_wireframe(one_over_eta_mesh_grid, xi_mesh_grid*100, theta_vals_results.T*100)

ax.plot_wireframe(one_over_eta_mesh_grid, xi_mesh_grid*100, c_vals_opt_k_results)

# Plot the curve showing the relationship between one_over_eta and xi_star at the bottom
#ax.plot(one_over_eta_grid, xi_star_results*100, -np.array(ln_m_H_star_results), color='r', marker='o', markersize=8, linewidth=3, label=r'Private Asset SAA ($\xi^*$)')

# Set labels and z-axis limit with increased font size
ax.set_xlabel(r'Expected waiting time, ($1/\eta$)', fontsize=15)
ax.set_ylabel(r'Allocation to Private Asset, ($\xi$) (%)', fontsize=15)
ax.set_zlabel(r'$-ln(-H(\xi)k$', fontsize=14)

# Rotate the surface to the left and ensure z-label is readable
ax.view_init(elev=20., azim=40)
ax.zaxis.labelpad = 20

# Increase font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=14)

# Make layout tight and place legend at the bottom
plt.tight_layout()
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=12)

saveFig('cFn3d') #'valueFn3d'
#plt.show()

# %% 3D Histogram plot

# Convert lists to numpy arrays for easier manipulation
xi_sim_results = np.array(xi_sim_results)

# Create a grid for eta values and xi_sim values
one_over_eta_mesh_grid, xi_mesh_grid = np.meshgrid(one_over_eta_grid, model.Xi_t)

# Flatten the data for histogram plotting
xAmplitudes = xi_sim_results.flatten() * 100
yAmplitudes = np.repeat(one_over_eta_grid, xi_sim_results.shape[1])

x = np.array(xAmplitudes)   # turn x,y data into numpy arrays
y = np.array(yAmplitudes)

fig = plt.figure(figsize=(10,8))  # create a canvas, tell matplotlib it's 3d
ax = fig.add_subplot(111, projection='3d')

# make histogram stuff - set bins - I choose 20x20 because I have a lot of data
hist, xedges, yedges = np.histogram2d(x, y, bins=(20,20))
xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])

xpos = xpos.flatten() / 2.
ypos = ypos.flatten() / 2.
zpos = np.zeros_like(xpos)

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
dz = hist.flatten()

cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
max_height = np.max(dz)  # get range of colorbars so we can normalize
min_height = np.min(dz)
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k - min_height) / max_height) for k in dz]

ax.bar3d(ypos, xpos, zpos, dy, dx, dz, color=rgba, zsort='average')
plt.title("3D Histogram of xi_sim_results")
ax.set_xlabel(r'Expected waiting time to trade ($1/\eta$)', fontsize=14)
ax.set_ylabel(r'Allocation Private Asset ($\xi$) (%)', fontsize=14)
ax.set_zlabel('Frequency', fontsize=14)

plt.tight_layout()
plt.show()


# %% CREATING KERNEL PLOTS 

# Create a kernel density plot for the first three rows of xi_sim_results
plt.figure(figsize=(5, 4))

colors = ['blue', 'green', 'red']
line_styles = ['-', '--', '-.']
labels = ['5-year','15-year']

for i in range(2):
    sns.kdeplot(xi_sim_results[i], color=colors[i], label=labels[i],linestyle=line_styles[i])
    plt.axvline(x=xi_star_results[i], color=colors[i], linestyle=':')

plt.xlabel(r'Allocation Private Asset ($\xi$)',fontsize=12)
plt.ylabel('Density',fontsize=12)
#plt.title('Kernel Density Plot of xi_sim_results')
plt.legend(fontsize=14)
plt.xlim(0, 1)

plt.tight_layout()
saveFig('kernelPlot_lp')

plt.show()


