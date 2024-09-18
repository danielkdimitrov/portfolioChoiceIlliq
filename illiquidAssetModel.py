import numpy as np
from numpy.polynomial.hermite import hermgauss
from itertools import product
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
        self.N = len(self.mu)  # Number of risky assets
        self.mu_w = mu[:-1]
        self.mu_x = mu[-1]
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
        self.nodes, self.weights = self.generate_quadrature_points(self.N)
    
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
    
    def generate_quadrature_points(self, N, n_points=10):
        """Generate quadrature points and weights for multivariate integration."""
        # draw quadrature nodes (x) and weights (w) in 1D 
        x, w = hermgauss(quad_points)
        
        # Constant related to Gauss-Hermite normalization
        self.const = 1/ np.pi**(self.N/2)

        # Multidimensional Gauss-Hermite quadrature setup
        xn = np.array(list(itertools.product(*(x,) * self.N)))
        self.wn = np.prod(np.array(list(itertools.product(*(w,) * self.N))), axis=1)
        # transformed 
        self.yn = np.sqrt(2.) * np.dot(self.sigma, xn.T).T + mu[None, :]
        
        return nodes_Nd, weights
    
#    def transformed_nodes(self):
#        """Transform the quadrature nodes to match the distribution of the random variables."""
#        return self.mu + np.dot(self.nodes, self.sigma.T) * np.sqrt(2)

    def wealth_growth(self, theta_t, c_t, xi_t, dZ):
        """
        Evaluate the growth rates R_{w,t + \Delta t}, R_{x,t + \Delta t}, and R_{q,t + \Delta t}.
        Note that we multiply by sqrt(2) to accomodate the quadrature transformation
        """
        # Liquid asset growth
        R_w = 1 + (self.r + theta_t @ (self.mu_w - self.r) - c_t) * self.dt + theta_t @ self.sigma_w @ dZ * np.sqrt(self.dt)*np.sqrt(2
        
        # Illiquid asset growth
        R_x = 1 + self.mu_x * self.dt + self.sigma_x @ dZ * np.sqrt(self.dt)*np.sqrt(2)
        
        # Total wealth growth
        R_q = (1 - xi_t) * R_w + xi_t * R_x
        
        # Change in illiquid asset share
        xi_next = xi_t * (R_x / R_q)
        
        return R_q, xi_next
    
    def expectation(self, transformed_nodes):
        """
        Calculate the expectation of a function using Hermite-Gauss quadrature.
        
        Parameters:
        xi_t (float): Current illiquid asset share.
        theta_t (np.array): Vector of weights.
        c_t (float): Consumption rate.
        function (callable): Function for which to calculate the expectation.
        
        Returns:
        float: The calculated expectation.
        """
        #transformed_nodes = self.transformed_nodes()
        #expectations = []
        
        #for dZ in transformed_nodes:
        #    _, _, R_q, xi_next = self.wealth_growth(theta_t, c_t, xi_t, dZ)
        #    expectations.append(function(R_q))  # Apply function to R_q
        
        # Compute the expectation as the weighted sum
        expectation = np.sum(self.weights * transformed_nodes) / np.pi**(self.N/2)
        return expectation
    
    def bellman_equation(self, xi_t, H_func, H_star):
        'TODO : integrate th'
        R_q, xi_next = wealth_growth(self, theta_t, c_t, xi_t, dZ)
        """Solve the Bellman equation."""
        def objective(theta_t, c_t, dZ):
            # Calculate the terms in the Bellman equation
            util = self.utility(c_t * (1 - xi_t)) 
            H_next_liq =  H_star * self.expectation(R_q**(1 - self.gamma))
            H_next_illiq = self.expectation(R_q**(1 - self.gamma)) *  H_func(xi_next))
            
            return util* self.dt + self.delta *(self.p * H_next_liq + (1 - self.p) *H_next_illiq)
        
        # Create a lambda function that binds dZ to the objective function
        objective_with_dZ = lambda theta_t, c_t: objective(theta_t, c_t, dZ)
        # Optimization over theta_t and c_t (could be done using scipy.optimize or any optimizer)
        'TODO: Write the optimizer'
        H_t, optimal_theta_t, optimal_c_t = self.optimize(objective_with_dZ)
        return H_t, optimal_theta_t, optimal_c_t
    
    def optimize(self, objective):
        """Optimize the Bellman equation (dummy implementation for illustration)."""
        # In practice, use scipy.optimize or other methods to find the optimal theta_t and c_t.
        # This is a placeholder function.
        H_t = 100
        theta_t_opt = np.ones(self.mu_w.shape[0]) / self.mu_w.shape[0]  # Dummy equal weighting
        c_t_opt = 0.02  # Dummy consumption rate
        return H_t, theta_t_opt, c_t_opt
    
    def solve(self, xi_0, H_func, H_star):
        """Solve the dynamic problem."""
        xi_t = xi_0
        for t in range(100):  # Number of time steps (this is an example, adjust as needed)
            optimal_theta_t, optimal_c_t = self.bellman_equation(xi_t, H_func, H_star)
            # Update state based on the dynamics
            _, _, _, xi_t = self.simulate_growth(optimal_theta_t, optimal_c_t, xi_t, np.random.normal(size=(self.N-1,)))
            print(f"At time {t}, xi_t = {xi_t}, optimal_c_t = {optimal_c_t}")

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
