# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:12:47 2024

@author: danie
"""

import numpy as np

class IlliquidAssetModel:
    def __init__(self, mu, Sigma, gamma, beta, p, r, dt):
        '''        

        Parameters
        ----------
        mu: Array; Vector of expected risky asset returns.
        Sigma : Array. Covariance matrix of the risky asset returns.
        gamma : Scalar. Risk aversion.
        beta : Scalar. 
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        r : TYPE
            DESCRIPTION.
        dt : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        self.mu_w = mu_w
        self.mu_x = mu_x
        self.sigma = cholesky(Sigma, lower=True)
        self.sigma_w = sigma_w
        self.sigma_x = sigma_x
        self.gamma = gamma  # Risk aversion coefficient
        self.delta = delta  # Discount factor
        self.p = p          # Probability of being able to trade
        self.r = r          # Risk-free rate
        self.dt = dt        # Time increment
    
    def utility(self, c):
        """CRRA utility function."""
        return c**(1 - self.gamma) / (1 - self.gamma)
    
    def simulate_growth(self, theta_t, c_t, xi_t):
        """Simulate the growth rates R_{w,t + \Delta t}, R_{x,t + \Delta t}, and R_{q,t + \Delta t}."""
        # Generate standard normal random variables
        delta_Z = np.random.normal(size=(self.sigma_w.shape[1],))

        # Liquid asset growth
        R_w = 1 + (self.r + theta_t @ (self.mu_w - self.r) - c_t) * self.dt + theta_t @ self.sigma_w @ delta_Z * np.sqrt(self.dt)
        
        # Illiquid asset growth
        R_x = 1 + self.mu_x * self.dt + self.sigma_x @ delta_Z * np.sqrt(self.dt)
        
        # Total wealth growth
        R_q = (1 - xi_t) * R_w + xi_t * R_x
        
        # Change in illiquid asset share
        xi_next = xi_t * (R_x / R_q)
        
        return R_w, R_x, R_q, xi_next

    def bellman_equation(self, xi_t, H_func, H_star):
        """Solve the Bellman equation."""
        def objective(theta_t, c_t):
            # Generate quadrature points and weights
            
            # Simulate the next period's growth rates
            R_w, R_x, R_q, xi_next = self.simulate_growth(theta_t, c_t, xi_t, delta_Z)
            
            # Calculate the terms in the Bellman equation
            term_1 = self.utility(c_t * (1 - xi_t)) * self.dt
            term_2 = self.delta * self.p * H_star * np.mean(R_q**(1 - self.gamma))
            term_3 = self.delta * (1 - self.p) * np.mean((R_q**(1 - self.gamma)) * H_func(xi_next))
            
            return term_1 + term_2 + term_3
        
        # Optimization over theta_t and c_t (could be done using scipy.optimize or any optimizer)
        optimal_theta_t, optimal_c_t = self.optimize(objective)
        return optimal_theta_t, optimal_c_t
    
    def optimize(self, objective):
        """Optimize the Bellman equation (dummy implementation for illustration)."""
        # In practice, use scipy.optimize or other methods to find the optimal theta_t and c_t.
        # This is a placeholder function.
        theta_t_opt = np.ones(self.mu_w.shape[0]) / self.mu_w.shape[0]  # Dummy equal weighting
        c_t_opt = 0.02  # Dummy consumption rate
        return theta_t_opt, c_t_opt
    
    def solve(self, xi_0, H_func, H_star):
        """Solve the dynamic problem."""
        xi_t = xi_0
        for t in range(100):  # Number of time steps (this is an example, adjust as needed)
            optimal_theta_t, optimal_c_t = self.bellman_equation(xi_t, H_func, H_star)
            # Update state based on the dynamics
            _, _, _, xi_t = self.simulate_growth(optimal_theta_t, optimal_c_t, xi_t)
            print(f"At time {t}, xi_t = {xi_t}, optimal_c_t = {optimal_c_t}")

# Example usage:

# Define parameters
mu_w = np.array([0.05, 0.06])
mu_x = 0.04
sigma_w = np.array([[0.1, 0.05], [0.05, 0.15]])
sigma_x = np.array([0.08, 0.07])
gamma = 2.0
delta = 0.95
p = 0.1
r = 0.03
dt = 0.01

# Initialize the model
model = IlliquidAssetModel(mu_w, mu_x, sigma_w, sigma_x, gamma, delta, p, r, dt)

# Dummy functions for H and H_star (replace with actual functions)
H_func = lambda xi: xi**(1 - gamma)
H_star = 1.0

# Initial value of xi
xi_0 = 0.5

# Solve the dynamic problem
model.solve(xi_0, H_func, H_star)
