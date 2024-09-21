import numpy as np
from numpy.polynomial.hermite import hermgauss
import itertools
from scipy.linalg import cholesky
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, minimize_scalar


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
        self.Xi_t = np.linspace(0.01,.99, self.gridpoints_Xi)
        
        # initialize
        self.H_t_vals_opt  =  self.utility(.1*self.Xi_t)
        self.H_func = UnivariateSpline(self.Xi_t, self.H_t_vals_opt, s=0)
        self.H_star = max(self.H_t_vals_opt)
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

    
    def H_t_objective(self, xi_t, params):
        theta_t, c_t = params[:-1], params[-1]
        #get next period's dynamic
        R_q, xi_next = self.wealth_growth(theta_t, c_t, xi_t)
        'Calculate the terms in the Bellman equation'
        util = self.utility(c_t * (1 - xi_t))         # get utility
        H_next_liq = self.H_star * self.expectation(R_q**(1 - self.gamma))         # get terms in case of liquidity
        H_next_illiq = self.expectation(R_q**(1 - self.gamma) * self.H_func(xi_next))          # get terms in case of illiquidity
        H_t_val = util * self.dt + self.delta * (self.p * H_next_liq + (1 - self.p) * H_next_illiq)

        # Debugging: Print objective function value
        print(f"ln(-H_t_val): {np.log(-H_t_val)}")

        return H_t_val  

    def bellman_equation(self, xi_t):
        '''
        optimize the Bellman equation in a given run.  
        ----------
        xi_t : Scalar. Current share of illiquid wealth.
        H_func : Function. Spline interpolator function for the H function.

        '''
        #optimize the Bellman equation given a xi_t
        objective = lambda params: np.log(-self.H_t_objective(xi_t, params)) # Minimize negative of value function (for optimization)
        init_guess = np.append(0.2 * (1-xi_t)* np.ones(self.mu_w.shape[0]), 0.02*(1-xi_t))  # Initial guess for theta and c
        #bounds = [(0, 1) for _ in range(self.mu_w.shape[0])] + [(0, 1)]  # Bounds for optimization
        result = minimize(objective, init_guess,method='L-BFGS-B') #, bounds=bounds
        theta_t_opt, c_t_opt = result.x[:-1], result.x[-1]
        
        H_t_val_opt = -np.exp(result.fun)  # Get the maximum value of the function (negative of objective)
        return H_t_val_opt, theta_t_opt, c_t_opt
    

    def solve(self, tol=1e-6, max_iter=100):
        # Store the optimal controls and value function
        self.optimal_theta = np.zeros((max_iter, self.gridpoints_Xi, len(self.mu_w)))
        self.optimal_consumption = np.zeros((max_iter, self.gridpoints_Xi))
        H_t_vals_opt_k = np.zeros_like(self.Xi_t)
        
        for k in range(max_iter):
            H_vals = np.zeros(self.gridpoints_Xi)
            for j, xi_j in enumerate(self.Xi_t):
                print(f'\n Current xi={xi_j}')
                H_t_val_opt, theta_opt, c_opt = self.bellman_equation(xi_j)
                # Update 
                H_t_vals_opt_k[j] = H_t_val_opt
                self.optimal_theta[k, j, :] = theta_opt
                self.optimal_consumption[k, j] = c_opt
            
            # Compute the error between current and previous value functions
            error = np.linalg.norm(np.log(self.H_func(self.Xi_t)) - np.log(H_t_vals_opt_k))
            # Update. Fit a new cubic spline based on updated H values
            self.H_func = UnivariateSpline(self.Xi_t, H_vals, s=0)
            self.H_star = max(H_t_vals_opt_k)

             # Print every 10th iteration
            if k % 10 == 0:
                print(f"Iteration {k}: Error = {error:.6f}")
            if error < tol:
                print(f"Converged in {k+1} iterations.")
                break
        else:
            print("Failed to converge within the maximum iterations.")
    

# Example usage:
# Define parameters
mu = np.array([0.055, 0.055])  # Example: two liquid assets and one illiquid asset
Sigma = np.array([[0.14**2,0.], [0.,0.14**2]])
gamma = 6.0
beta = 0.03
eta = 0.1
r = 0.02
dt = 0.01

# Initialize the model
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model.solve()


# Solve the dynamic problem
#model.solve(xi_0, H_func, H_star)
