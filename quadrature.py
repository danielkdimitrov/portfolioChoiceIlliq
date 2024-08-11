# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:43:56 2024

@author: danie
"""

import numpy as np
from scipy.linalg import cholesky
from numpy.polynomial.hermite import hermgauss
import unittest


class NDGaussHermiteQuadrature:
    def __init__(self, mu, Sigma, func, n_points=10):
        self.mu = mu
        self.sigma = Sigma
        self.func = func
        self.n_points = n_points
        
        # draw quadrature nodes and weights in 1D 
        self.nodes, self.weights = hermgauss(n_points)
        self.L = cholesky(Sigma, lower=True)
    
    def integrate(self):
        d = len(self.mu)
        result = 0.0
        
        # for each combination in quadrature points find the trasformed nodes, multiply by corresponding weight, and keep the sum
        for indices in np.ndindex(*(self.n_points,) * d):
            weight_prod = np.prod([self.weights[i] for i in indices])
            nodes_prod = np.array([self.nodes[i] for i in indices])
            
            transformed_nodes = np.sqrt(2) * self.L @ nodes_prod + self.mu
            
            f_value = self.func(transformed_nodes)
            
            result += weight_prod * f_value
        
        result *= (1 / np.pi**(d / 2))
        
        return result

class TestNDGaussHermiteQuadrature(unittest.TestCase):
    def test_lognormal_expectation(self):
        # Define the mean vector and covariance matrix
        np.random.seed(0)
        n_vars = 2
        mean = 0.1
        sigma = 0.2
        rho = .5
        mu = np.full(n_vars, mean)
        #offdiagonal terms
        sigma = np.array([[sigma, 0],[rho*sigma,np.sqrt(1-rho**2)*sigma]])  #np.full((n_vars, n_vars), 0.1)
        # cov matrix
        Sigma = sigma @ sigma.T  # Ensure positive semi-definite matrix
        
        #portfolio weights, equally-weighted
        pi = np.full(n_vars, 1./n_vars)
        
        # Portfolio mean return
        mu_p = pi.T @ mu
        
        # Portfolio varaince
        variance_p =  pi.T @ Sigma @ pi

        # Analytical expected value for log-normal distribution 
        analytical_expectation = np.exp(mu_p + variance_p / 2)
        
        # Define the log-normal function
        def log_normal_function(x):
            return np.exp(x).sum()
        def sum_function(x):
            pi = np.array([.5, .5])
            return np.exp(np.dot(pi, x))
        
        
        # Create an instance of the NDGaussHermiteQuadrature class
        quadrature = NDGaussHermiteQuadrature(mu, Sigma, sum_function, n_points=10)
        
        # Perform the integration
        quadrature_expectation = quadrature.integrate()
        
        # Calculate the discrepancy
        discrepancy = np.abs(analytical_expectation - quadrature_expectation)
        
        print(f"Analytical expected value: {analytical_expectation}")
        print(f"Quadrature expected value: {quadrature_expectation}")
        print(f"Discrepancy: {discrepancy}")
        
        # Assert the discrepancy is within a reasonable tolerance
        self.assertAlmostEqual(analytical_expectation, quadrature_expectation, places=2)

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
