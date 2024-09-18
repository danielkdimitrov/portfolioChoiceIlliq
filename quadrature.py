# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:43:56 2024

@author: danie

Unit test for the quadrature method
"""

import numpy as np
from scipy.linalg import cholesky
import itertools
from numpy.polynomial.hermite import hermgauss
import unittest


class NDGaussHermiteQuadrature:
    def __init__(self, mu, Sigma, quad_points =10):
        self.mu = mu
        self.N = len(mu)
        
        # draw quadrature nodes (x) and weights (w) in 1D 
        x, w = hermgauss(quad_points)
        # cholesky decomposition of the cov matrix, with the added transform const
        self.sigma = cholesky(Sigma, lower=True)
        
        # Constant related to Gauss-Hermite normalization
        self.const = 1/ np.pi**(self.N/2)

        # Multidimensional Gauss-Hermite quadrature setup
        xn = np.array(list(itertools.product(*(x,) * self.N)))
        self.wn = np.prod(np.array(list(itertools.product(*(w,) * self.N))), axis=1)
        # transformed
        self.yn = np.sqrt(2.) * np.dot(self.sigma, xn.T).T + mu[None, :]
        
        # Calculate the expectation 
        #self.expectation = self.integrate(func)

    
    def integrate(self,func):        
        # for each combination in quadrature points find the trasformed nodes, multiply by corresponding weight, and keep the sum
        result = self.const*np.sum(self.wn*func(self.yn))
        
        return result

class TestNDGaussHermiteQuadrature(unittest.TestCase):
    def test_lognormal_expectation(self):
        # Define the mean vector and covariance matrix
        np.random.seed(0)
        n_vars = 2
        mean = 0.1
        r = 0.02
        sigma = 0.5
        rho = .5
        
        mu = np.full(n_vars, mean)
        #offdiagonal terms
        sigma = np.array([[sigma, 0],[rho*sigma,np.sqrt(1-rho**2)*sigma]])  #np.full((n_vars, n_vars), 0.1)
        # cov matrix
        Sigma = sigma @ sigma.T  # Ensure positive semi-definite matrix
        
        #portfolio weights, equally-weighted
        pi = np.full(n_vars, 1./(n_vars+1))
        
        # Portfolio mean return
        ones = np.ones_like(pi)
        mu_p =  pi.T @ mu + (1 - pi.T @ ones )* r 
        # Portfolio varaince
        variance_p =  pi.T @ Sigma @ pi

        # Analytical expected value for log-normal distribution 
        analytical_expectation = np.exp(mu_p + variance_p / 2)
        
        # Define the log-normal function
        def log_normal_portfolio_return(pi, Return, r):
            # Portfolio return
            #assetReturn = np.exp((1 - pi @  np.ones_like(pi))*r+ pi* Return)
            pReturn = (1 - pi @  np.ones_like(pi))*r + Return@pi.T
            exppReturn = np.exp(pReturn)
            return exppReturn 
        
        myLog_normal_portfolio_return = lambda Return : log_normal_portfolio_return(pi, Return, r)
                
        
        # Create an instance of the NDGaussHermiteQuadrature class
        myQuadrature = NDGaussHermiteQuadrature(mu, Sigma, 20)
        
        # Perform the integration 
        quadrature_expectation = myQuadrature.integrate(myLog_normal_portfolio_return)
        
        # Calculate the discrepancy
        discrepancy = np.abs(analytical_expectation - quadrature_expectation)
        
        print(f"Analytical expected value: {analytical_expectation}")
        print(f"Quadrature expected value: {quadrature_expectation}")
        print(f"Discrepancy: {discrepancy}")
        
        # Assert the discrepancy is within a reasonable tolerance
        self.assertAlmostEqual(analytical_expectation, quadrature_expectation, places=5)

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
