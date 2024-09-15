# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:43:56 2024

@author: danie
"""

import numpy as np
from scipy.linalg import cholesky
import itertools
from numpy.polynomial.hermite import hermgauss
import unittest


class NDGaussHermiteQuadrature:
    def __init__(self, mu, Sigma, quad_points =10):
        self.mu = mu
        self.sigma = Sigma
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
        #d = len(self.mu) #number of risky assets
        #result = 0.0
        
        # for each combination in quadrature points find the trasformed nodes, multiply by corresponding weight, and keep the sum
        result = self.const*np.sum(self.wn*func(self.yn))
        #np.sum((self.wn * self.const)[:, None] * func(self.yn), axis=0)
        
        # 
        #
        #sself.const*self.quadrature_weights*self.f(quadrature_nodes)
        
        # for indices in np.ndindex(*(self.n_points,) * d):
        #     weight_prod = np.prod([self.weights[i] for i in indices])
        #     nodes_prod = np.array([self.nodes[i] for i in indices])
            
        #     transformed_nodes = np.sqrt(2) * self.sigma @ nodes_prod + self.mu
            
        #     f_value = self.func(transformed_nodes)
            
        #     result += weight_prod * f_value
        
        # result *= (1 / np.pi**(d / 2))
        
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
        print('Analytical:', analytical_expectation)
        
        # Define the log-normal function
        def log_normal_portfolio_return(pi, Return):
            # Portfolio return
            assetReturn = pi* np.exp(Return)    
            pReturn = assetReturn.sum(axis=1)
            return pReturn
        
        myLog_normal_portfolio_return = lambda Return : log_normal_portfolio_return(pi, Return)
        
        #def sum_function(x):
        #    pi = np.array([.5, .5])
        #    return np.exp(np.dot(pi, x))
        
        
        # Create an instance of the NDGaussHermiteQuadrature class
        myQuadrature = NDGaussHermiteQuadrature(mu, Sigma)
        
        # Perform the integration 
        expectation = myQuadrature.integrate(myLog_normal_portfolio_return)
        
        print(expectation)
        '''
        # Calculate the discrepancy
        discrepancy = np.abs(analytical_expectation - quadrature_expectation)
        
        print(f"Analytical expected value: {analytical_expectation}")
        print(f"Quadrature expected value: {quadrature_expectation}")
        print(f"Discrepancy: {discrepancy}")
        
        # Assert the discrepancy is within a reasonable tolerance
        self.assertAlmostEqual(analytical_expectation, quadrature_expectation, places=2)
        '''

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
