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
        #print(xi_next[xi_next>=1])
        # Debugging: Print objective function value
        #print(f"try c_t: {c_t}")
        #print(f"ln(-H_t_val): {np.log(-H_t_val)}")

        return H_t_val  

    def h_t_objective(self, xi_t, params):
        '''
        Returns the log-transform of the value function such that 
        h(xi) = np.log(-H(xi))  <=>  H(xi) = -np.exp(h(xi))
        '''
        
        theta_t, c_t = params[:-1], params[-1]
        #get next period's dynamic
        R_q, xi_next = self.wealth_growth(theta_t, c_t, xi_t)
        'Calculate the terms in the Bellman equation'
        util = self.utility(c_t * (1 - xi_t))         # get utility
        'Und to log transformation of the value function'
        H_star = -np.exp(self.h_star)
        H_illiq = -np.exp(self.h_func(xi_next))
        'Get next period bellman value'
        RHS_liq = H_star * self.expectation(R_q**(1 - self.gamma))         # get terms in case of liquidity
        RHS_illiq = self.expectation(R_q**(1 - self.gamma) * H_illiq)          # get terms in case of illiquidity
        H_t_val = util * self.dt + self.delta * (self.p * RHS_liq + (1 - self.p) * RHS_illiq)
        'Log-tranform again the value function'
        h_t_val = np.log(-H_t_val)
        print(f'Try theta={theta_t} and c={c_t}, get h={h_t_val}')        
        #print(xi_next[xi_next>=1])
        # Debugging: Print objective function value
        #print(f"try c_t: {c_t}")
        #print(f"ln(-H_t_val): {np.log(-H_t_val)}")
        return h_t_val  

    def bellman_equation(self, xi_t):
        '''
        optimize the Bellman equation in a given run.  
        ----------
        xi_t : Scalar. Current share of illiquid wealth.
        H_func : Function. Spline interpolator function for the H function.

        '''
        #optimize the Bellman equation given a xi_t
        objective = lambda params: self.h_t_objective(xi_t, params) # Minimize negative of value function (for optimization)
        #bounds = [(0, 1) for _ in range(self.mu_w.shape[0])] + [(0, 1)]  # Bounds for optimization
        init_guess = np.append(0.2 * (1-xi_t)* np.ones(self.mu_w.shape[0]), 0.03*(1-xi_t))  # Initial guess for theta and c
        bounds = [(0, 1) for _ in range(self.mu_w.shape[0])] + [(1e-4, 1)]  # Prevent consumption from being zero
        result = minimize(objective, init_guess, method='L-BFGS-B', bounds=bounds)
        theta_t_opt, c_t_opt = result.x[:-1], result.x[-1]
        if result.success == False: print(f"Optimization convergence: {result.success}")
        
        H_t_val_opt = -np.exp(-result.fun)  # Get the maximum value of the function (negative of objective)
        print(f'h^* =: {H_t_val_opt}')
        return H_t_val_opt, theta_t_opt, c_t_opt
        
    def solve(self, tol=1e-6, max_iter=250):
        # Store the optimal controls and value function
        self.theta_opt = np.zeros((self.gridpoints_Xi, len(self.mu_w)))
        self.c_opt = np.zeros(self.gridpoints_Xi)
        
        # initialize with the Merton solutions
        h_t_vals_opt_k =  np.log(-self.H_m) *np.ones_like(self.Xi_t) 
        #H_t_vals_opt_k = np.zeros_like(H_t_vals_opt) #the new points
        self.h_func = UnivariateSpline(self.Xi_t, h_t_vals_opt_k, s=0)
        self.h_star = h_t_vals_opt_k[0]

        # enable interactive mode
        #plt.ion()
        axs = None  # initialize axis
        for k in range(max_iter):
            #print(f'\n ----- Iteration k={k} ---------------')
            for j, xi_j in enumerate(self.Xi_t):
                #print(f'Current xi={xi_j}')
                h_t_val_opt, self.theta_opt[j,:], self.c_opt[j] = self.bellman_equation(xi_j)
                # Update 
                h_t_vals_opt_k[j] = h_t_val_opt
                #self.init_guess = np.array((self.theta_opt[j, :][0], self.c_opt[j]))
            
            # Compute the error between current and previous value functions
            error = np.linalg.norm(self.h_func(self.Xi_t) - h_t_vals_opt_k)
            # Update. Fit a new cubic spline based on updated H values
            #self.H_func = UnivariateSpline(self.Xi_t, H_t_vals_opt_k, s=0)
            'Update the fitted value function'
            self.h_func = UnivariateSpline(self.Xi_t, h_t_vals_opt_k, s=0)
            self.h_star = max(self.h_func(np.linspace(.001, .99, 250)))
             # Print every k-th iteration
            if k % 10 == 0:
                print(f"Iteration {k}: Error = {error:.6f}")
                #plot the new value function
                #plt.plot(self.Xi_t, -np.log(-H_t_vals_opt_k))
                #plt.plot(self.Xi_t, self.theta_opt)
                axs = self.plot_results(axs)

            if error < tol:
                print(f"Converged in {k+1} iterations.")
                break
        else:
            print("Failed to converge within the maximum iterations.")
            
            
    def plot_results(self, axs=None):
        """
        Plot value function, optimal consumption, and optimal portfolio weights.
        """
        
        if axs is None:
            # Initialize a 3x1 subplot grid only once
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column

        # Clear previous data before plotting new ones in the same loop
        #axs[0].cla()
        #axs[1].cla()
        #axs[2].cla()

    	# Plot log(-H_t_vals_opt_k) in the first subplot
        axs[0].plot(self.Xi_t, -self.h_func(self.Xi_t))
        axs[0].set_title("Value Function: -log(-H_t_vals_opt_k)")
        axs[0].set_xlabel("xi_t")
        axs[0].set_ylabel("log(-H)")
	    
        # Plot consumption (c_opt) in the second subplot
        axs[1].plot(self.Xi_t, self.c_opt)
        axs[1].set_title("Optimal Consumption (c_opt)")
        axs[1].set_xlabel("xi_t")
        axs[1].set_ylabel("Consumption (c_t)")
	    
        # Plot theta_opt in the third subplot (can be multiple lines if self.theta_opt has multiple dimensions)
        for i in range(self.theta_opt.shape[1]):
            axs[2].plot(self.Xi_t, self.theta_opt[:, i], label=f"Theta {i+1}")
            axs[2].set_title("Optimal Portfolio Weights (theta_opt)")
            axs[2].set_xlabel("xi_t")
            axs[2].set_ylabel("theta_t")
		    
    	# Adjust layout
        plt.tight_layout()
    
        # Pause briefly to update the plot in interactive mode
        #plt.pause(0.01)
    	# Show the plot
            #plt.show()

# Example usage:
# Define parameters
mu = np.array([0.055, 0.055])  # Example: two liquid assets and one illiquid asset
Sigma = np.array([[0.14**2,0.], [0.,0.14**2]])
gamma = 6.0
beta = 0.03
eta = 1/10
r = 0.02
dt = 1.

# Initialize the model
model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
model.solve()


# Solve the dynamic problem
#model.solve(xi_0, H_func, H_star)
