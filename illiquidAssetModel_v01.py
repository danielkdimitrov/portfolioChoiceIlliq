import numpy as np
from numpy.polynomial.hermite import hermgauss
import itertools
from scipy.linalg import cholesky
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt  

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
        
        self.Sigma = Sigma
        self.sigma = cholesky(Sigma, lower=True)
        self.sigma_w = self.sigma[:-1, :]
        self.sigma_x = self.sigma[-1, :]
        self.r = r  # Risk-free rate
        
        # Utility model
        self.beta = beta
        self.gamma = gamma  # Risk aversion coefficient
        self.delta = np.exp(-beta * dt)  # Discount factor
        self.dt = dt  # Time increment
        
        # Trading probability
        self.p = self.trading_probability(eta)
        
        # Merton solution 
        self.pi_m, self.c_m, self.H_m = self.merton_solution()
        
        # Generate quadrature points and weights
        self.xn, self.Wn = self.generate_quadrature_points()
        
        # Create a grid over xi
        self.gridpoints_Xi = 20
        self.Xi_t = np.linspace(0.01,.99, self.gridpoints_Xi)
        
    def merton_solution(self):
        """
        Provides the closed-form solution for the Merton optimal allocation and consumption.
        
        Returns
        -------
        pi_opt : array
            Optimal investment allocation vector.
        c_opt : float
            Optimal consumption.
        lambda_ : array
            Market price of risk.
        """
        # Compute the market price of risk: lambda = sigma^{-1} (mu - r1)
        mu_minus_r = self.mu - self.r * np.ones(len(self.mu))
        lambda_ = np.linalg.solve(self.sigma, mu_minus_r)
        
        # Optimal portfolio allocation: pi = (1 / gamma) * Sigma^{-1} * (mu - r1)
        pi_opt = (1 / self.gamma) * np.linalg.inv(self.Sigma) @ mu_minus_r
                
        # Optimal consumption: c = (beta + r(gamma - 1)) / gamma + 0.5 * (gamma - 1) / gamma^2 * ||lambda||^2
        c_opt = (self.beta + self.r * (self.gamma - 1)) / self.gamma + 0.5 * (self.gamma - 1) / (self.gamma**2) * np.dot(lambda_, lambda_)
        
        # Get the reduced value scalar for Merton 
        H = (1/(1-self.gamma))*(1/c_opt)**self.gamma
        
        return pi_opt, c_opt, H        
        
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
        if c <= 0:
            return -np.inf  # Penalize zero or negative consumption
        elif self.gamma == 1:
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
        #print(f"try c_t: {c_t}")
        #print(f"ln(-H_t_val): {np.log(-H_t_val)}")

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
        #bounds = [(0, 1) for _ in range(self.mu_w.shape[0])] + [(0, 1)]  # Bounds for optimization
        init_guess = np.append(0.2 * (1-xi_t)* np.ones(self.mu_w.shape[0]), 0.02*(1-xi_t))  # Initial guess for theta and c        
        result = minimize(objective, init_guess,method='Nelder-Mead') #, bounds=bounds
        theta_t_opt, c_t_opt = result.x[:-1], result.x[-1]
        if result.success == False: print(f"Optimization convergence: {result.success}")
        
        H_t_val_opt = -np.exp(result.fun)  # Get the maximum value of the function (negative of objective)
        return H_t_val_opt, theta_t_opt, c_t_opt

    def plot_results(self, H_t_vals_opt_k, axs=None, lines=None, iteration=None):
        """
        Update the value function, optimal consumption, and portfolio weights in the same plot.
        The behavior is equivalent to MATLAB's 'hold on'.
        """
        if axs is None:  # Initialize subplots only once
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))
            
            # Create empty line objects that will be updated
            lines = {
                'H_t_vals': axs[0].plot(self.Xi_t, -np.log(-H_t_vals_opt_k), label="log(-H)")[0],
                'c_opt': axs[1].plot(self.Xi_t, self.c_opt, label="c_opt")[0],
                'theta_opt': [axs[2].plot(self.Xi_t, self.theta_opt[:, i], label=f"Theta {i+1}")[0]
                              for i in range(self.theta_opt.shape[1])]
            }
            
            # Set titles and labels for subplots
            axs[0].set_title("Value Function: log(-H_t_vals_opt_k)")
            axs[0].set_xlabel("xi_t")
            axs[0].set_ylabel("log(-H)")
            
            axs[1].set_title("Optimal Consumption (c_opt)")
            axs[1].set_xlabel("xi_t")
            axs[1].set_ylabel("Consumption (c_t)")
            
            axs[2].set_title("Optimal Portfolio Weights (theta_opt)")
            axs[2].set_xlabel("xi_t")
            axs[2].set_ylabel("theta_t")
            
            # Add legends to all subplots
            for ax in axs:
                ax.legend()
            
            # Adjust layout for better visualization
            plt.tight_layout()
        
        else:
            # Update the y-data of the existing lines without creating new plots
            lines['H_t_vals'].set_ydata(-np.log(-H_t_vals_opt_k))
            lines['c_opt'].set_ydata(self.c_opt)
            for i, line in enumerate(lines['theta_opt']):
                line.set_ydata(self.theta_opt[:, i])
        
        # Redraw the figure to reflect updates
        plt.draw()
        plt.pause(0.01)  # Pause to ensure the plot is updated
        
        return axs, lines  # Return updated axes and line objects
    
    def solve(self, tol=1e-6, max_iter=500):
        # Store the optimal controls and value function
        self.theta_opt = np.zeros((self.gridpoints_Xi, len(self.mu_w)))
        self.c_opt = np.zeros(self.gridpoints_Xi)
    
        # Initialize with the Merton solutions
        H_t_vals_opt_k = self.H_m * np.ones_like(self.Xi_t)
        self.H_func = UnivariateSpline(self.Xi_t, H_t_vals_opt_k, s=0)
        self.H_star = H_t_vals_opt_k[0]
    
        # Enable interactive mode for live plotting
        plt.ion()
        axs = None  # Initialize axis object
        lines = None  # Initialize line objects
        
        for k in range(max_iter):
            for j, xi_j in enumerate(self.Xi_t):
                H_t_val_opt, self.theta_opt[j, :], self.c_opt[j] = self.bellman_equation(xi_j)
                H_t_vals_opt_k[j] = H_t_val_opt
    
            # Compute the error between current and previous value functions
            error = np.linalg.norm(np.log(-self.H_func(self.Xi_t)) - np.log(-H_t_vals_opt_k))
            self.H_func = UnivariateSpline(self.Xi_t, H_t_vals_opt_k, s=0)
            self.H_star = max(self.H_func(np.linspace(.001, .99, 250)))
    
            # Print every k-th iteration
            if k % 10 == 0:
                print(f"Iteration {k}: Value Fn Diff = {error:.6f}")
                #axs, lines = self.plot_results(H_t_vals_opt_k, axs, lines, iteration=k)
    
            # Stop if the error is below the tolerance
            if error < tol:
                print(f"Converged in {k+1} iterations.")
                break
        else:
            print("Failed to converge within the maximum iterations.")
    
        # Keep the plot open after convergence
        plt.ioff()
        plt.show()
        
    # Example usage:
# Define parameters
mu = np.array([0.055, 0.055])  # Example: two liquid assets and one illiquid asset
Sigma = np.array([[0.14**2,0.], [0.,0.14**2]])
gamma = 6.0
beta = 0.03
eta = 1/12
r = 0.02
dt = 1.

# Initialize the model
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model.solve()

# Evaluate the H_func over the grid Xi_t
H_vals_on_grid = model.H_func(model.Xi_t)

# Find the index where H_func equals H_star (or is closest to it)
index_of_H_star = np.argmax(H_vals_on_grid)

# Corresponding Xi_t value
xi_star = model.Xi_t[index_of_H_star]

print(f"The value of xi_t corresponding to H_star is: {xi_star}")
model.plot_results(H_vals_on_grid)
# Solve the dynamic problem
#model.solve(xi_0, H_func, H_star)
