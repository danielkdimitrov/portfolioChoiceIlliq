import numpy as np
from numpy.polynomial.hermite import hermgauss
import itertools
from scipy.linalg import cholesky

class IlliquidAssetModel:
    def __init__(self, mu, Sigma, gamma, beta, eta, r, dt):
        '''        
        Parameters
        ----------
        mu: Array; Vector of expected risky asset returns. 
            The last row refers to the illiquid asset. 
        Sigma : Array. Covariance matrix of the risky asset returns. 
            The last row refers to the illiquid asset. 
        gamma : Scalar. Risk aversion.
        beta : Scalar. Discount factor.
        eta : Scalar. Average waiting time to trade.
        r : Scalar. Interest Rate.  
        dt : Scalar. Time step.

        Returns
        -------
        None.
        '''
        self.mu = mu
        self.n = len(self.mu)  # Number of risk factors (corresponding to the num. of risky assets, incl. the illiquid one)

        self.mu_w = mu[:-1]
        self.mu_x = mu[-1]
        self.m = 10 #number of quadrature points 
        
        self.sigma = cholesky(Sigma, lower=True)
        self.sigma_w = self.sigma[:-1, :]
        self.sigma_x = self.sigma[-1, :]
        self.r = r  # Risk-free rate
        
        # Utility model
        self.gamma = gamma  # Risk aversion coefficient
        self.delta = np.exp(-beta * dt)  # Discount factor
        self.dt = dt  # Time increment
        
        # Trading probability
        self.p = self.trading_probability(eta)
        
        # Generate quadrature points and weights
        self.xn, self.Wn = self.generate_quadrature_points()
        
        # Create a grid over xi
        self.gridpoints_Xi = 20
        self.Xi_t = np.linspace(0,.99, gridpoints_Xi)
        
        # initialize
        H_str
        H0 =  
        
        
        # loop over each gridpoin on xi
            # fit cubic spline over H, start with the initialization
            # get the optimal H_str, c, theta, xi_str-> code the optimizer
            # evaluate the Bellman equation's RHS
            # update the values on the function H,

        
        
        # incert overarching loop over the value function improment
    
    def trading_probability(self, eta):
        """
        Computes the trading probability p given the parameter eta and the time increment dt.
        
        Parameters:
        eta (float): The parameter governing the probability.
        
        Returns:
        float: The trading probability p.
        """
        return 1 - np.exp(-eta * self.dt)
    
    def utility(self, c):
        """CRRA utility function."""
        if self.gamma == 1:
            return np.log(c)
        else:
            return c**(1 - self.gamma) / (1 - self.gamma)
    
    def generate_quadrature_points(self):
        """
        Generate quadrature points and weights for multivariate integration.
        Note that mean-variance transformation of the nodes is done at the wealth process functions
        """
        # draw quadrature nodes (x) and weights (w) in 1D 
        x, w = hermgauss(self.m)
        
        # Constant related to Gauss-Hermite normalization
        self.const = 1/ np.pi**(self.n/2)

        # Multidimensional Gauss-Hermite quadrature setup. Get the Cartesian product of N copies of the quadrature nodes and weights
        xn = np.array(list(itertools.product(*(x,) * self.n)))
        wn = np.array(list(itertools.product(*(w,) * self.n)))
        # get product of weights
        Wn = np.prod(wn, axis=1) 
        # get z variable (transformed nodes stacked, transpose the vertor row) 
        
        return xn, Wn
    
    def wealth_growth(self, theta_t, c_t, xi_t):
        """
        Evaluate the growth rates R_{w,t + \Delta t}, R_{x,t + \Delta t}, and R_{q,t + \Delta t}.
        Note that we multiply by sqrt(2) to accomodate the quadrature transformation
        """
        dZ = self.xn*np.sqrt(2)
        # Liquid asset growth
        R_w = 1 + (self.r + theta_t @ (self.mu_w - self.r) - c_t) * self.dt + (theta_t @ self.sigma_w) @ dZ.T * np.sqrt(self.dt)
        
        # Illiquid asset growth
        R_x = 1 + self.mu_x * self.dt + self.sigma_x @ dZ.T * np.sqrt(self.dt)
        
        # Total wealth growth
        R_q = (1 - xi_t) * R_w + xi_t * R_x
        
        # Change in illiquid asset share
        xi_next = xi_t * (R_x / R_q)
        
        return R_q, xi_next
    
    def expectation(self, transformed_nodes):
        """
        Calculate the expectation of a function using Hermite-Gauss quadrature.
        Parameters:   transformed_nodes (float): correspond to f(x_i) where x are the GH quardature nodes        
        Returns:      float: The calculated expectation.
        """
        expectation = self.const*np.sum(self.Wn * transformed_nodes)
        return expectation

    def bellman_equation(self, xi_t, H_func):
        'TODO : integrate th'

        #get optimal controls        
        H_str, theta_t_opt, c_t_opt =  self.optimize()
        xi_t = .1
        #get next period's dynamic
        R_q, xi_next = self.wealth_growth(theta_t_opt, c_t_opt, xi_t)

        # Calculate the terms in the Bellman equation
        # get utility
        util = self.utility(c_t_opt * (1 - xi_t))
        # get terms in case of liquidity
        H_next_liq =  H_star * self.expectation(R_q**(1 - self.gamma))
        # get terms in case of illiquidity
        H_next_illiq = self.expectation(R_q**(1 - self.gamma)* H_func(xi_next))
            
        H_t = util* self.dt + self.delta *(self.p * H_next_liq + (1 - self.p) *H_next_illiq)
        
        # Create a lambda function that binds dZ to the objective function
        objective_with_dZ = lambda theta_t, c_t: objective(theta_t, c_t, dZ)
        # Optimization over theta_t and c_t (could be done using scipy.optimize or any optimizer)
        'TODO: Write the optimizer'
        H_t, optimal_theta_t, optimal_c_t = self.optimize(objective_with_dZ)
        return H_t, optimal_theta_t, optimal_c_t

    
    def optimize(self):
        #, objective
        """Optimize the Bellman equation (dummy implementation for illustration)."""
        # TODO 
        # In practice, use scipy.optimize or other methods to find the optimal theta_t and c_t.
        # This is a placeholder function.
        H_str = 100
        theta_t_opt =.2*np.ones(self.mu_w.shape[0]) # Dummy equal weighting
        c_t_opt = 0.02  # Dummy consumption rate
        return H_str, theta_t_opt, c_t_opt
    '''
    def solve(self, xi_0, H_func, H_star):
        """Solve the dynamic problem."""
        xi_t = xi_0
        for t in range(100):  # Number of time steps (this is an example, adjust as needed)
            optimal_theta_t, optimal_c_t = self.bellman_equation(xi_t, H_func, H_star)
            # Update state based on the dynamics
            _, _, _, xi_t = self.simulate_growth(optimal_theta_t, optimal_c_t, xi_t, np.random.normal(size=(self.N-1,)))
            print(f"At time {t}, xi_t = {xi_t}, optimal_c_t = {optimal_c_t}")
    '''
# Example usage:
# Define parameters
mu = np.array([0.05, 0.06, 0.04])  # Example: two liquid assets and one illiquid asset
Sigma = np.array([[0.1, 0.05, 0.02], [0.05, 0.15, 0.03], [0.02, 0.03, 0.07]])
gamma = 2.0
beta = 0.95
eta = 0.1
r = 0.03
dt = 0.01

# Define the known H function
H_func = lambda xi: xi**(1 - gamma)  # Example of H function

# Initialize the model
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)

# Dummy value for H_star (you should replace this with the actual value)
H_star = 1.0

# Initial value of xi
xi_0 = 0.5

# Solve the dynamic problem
model.solve(xi_0, H_func, H_star)
