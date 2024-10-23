import numpy as np
from numpy.polynomial.hermite import hermgauss
import itertools
from scipy.linalg import cholesky
from scipy.interpolate import UnivariateSpline, CubicSpline, interp1d
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt  

class IlliquidAssetModel:
    def __init__(self, mu, Sigma, gamma, beta, eta, r, dt=1):
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
        
        # Merton solution, n assets 
        self.pi_m, self.c_m, self.H_m = self.merton_solution()
        self.cec_m = self.getCec(self.H_m)
        
        # Generate quadrature points and weights
        self.xn, self.Wn = self.generate_quadrature_points()
        
        # Create a grid over xi
        self.gridpoints_Xi = 20
        self.Xi_t = np.linspace(0.01,.99, self.gridpoints_Xi)
        
        # Finer grid 
        self.xi_fine_grid = np.linspace(.01, .99, 2000)
        
    def getCec(self, H):
        """
        Evaluates the certainty equivalent consumption (cec_xi) for a given xi_t.
        
        Parameters:
        xi_t : float
            The current value of xi (state variable).
        
        Returns:
        float
            The evaluated value of cec_xi.
        """        
        cec = (self.beta * (1 - self.gamma) * H) ** (1 / (1 - self.gamma))
        
        return cec
        
        
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
        Evaluate the growth rates R_{w,t + Delta t}, R_{x,t + Delta t}, and R_{q,t + Delta t}.
        We multiply by sqrt(2) to accomodate the quadrature transformation
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
        
        '''
        if c_t <= 0:
            print('Negative consumption')
        
        if np.any(xi_next > 1):
            mask = xi_next > 1
            print(f'xi_next> 0: {xi_next[mask]}')
            print(f'theta: {theta_t[0]:.4f}') 
        '''
        return R_q, xi_next
    
    def expectation(self, transformed_nodes):
        """
        Calculate the expectation of a function using Hermite-Gauss quadrature.
        Parameters:   transformed_nodes (float): correspond to f(x_i) where x are the GH quardature nodes        
        Returns:      float: The calculated expectation.
        """
        expectation = self.const*np.sum(self.Wn * transformed_nodes)
        return expectation

    
    def ln_m_H_t_objective(self, xi_t, params):
        theta_t, c_t = params[:-1], params[-1]
        #get next period's dynamic
        R_q, xi_next = self.wealth_growth(theta_t, c_t, xi_t)
        'Calculate the terms in the Bellman equation'
        if np.any(xi_next < 0) or np.any(xi_next > 1):  # This checks if any xi_next exceeds 1
            #H_next_illiq = -np.inf  #np.inf  # Enforce H_func going to negative infinity when xi_next > 1            
            c_t = 0.001
        
        # transform back the H_vals function from log minus
        H_vals_next = - np.exp(self.ln_m_H_func(xi_next))            
        # get terms in case of illiquidity
        H_next_illiq = self.expectation(R_q**(1 - self.gamma) * H_vals_next)          
        
        # get intermediate utility
        util = self.utility(c_t * (1 - xi_t))         
        
        # transform back ln minus H_star; get terms in case of liquidity
        H_next_liq = -np.exp(self.ln_m_H_star) * self.expectation(R_q**(1 - self.gamma))      
        
        # Get next period value fn
        H_t_val = util * self.dt + self.delta * (self.p * H_next_liq + (1 - self.p) * H_next_illiq)
        # Transform again into log minus value units 
        ln_m_H_t_val = np.log(-H_t_val)
        #print(f'ln_m_H_t_val={ln_m_H_t_val:.2f}')
        # Debugging: Print objective function value
        #print(f"try c_t: {c_t}")
        #print(f"ln(-H_t_val): {np.log(-H_t_val)}")

        return ln_m_H_t_val  

    def optimize(self, xi_t):
        '''
        optimize the Bellman equation in a given run.  
        ----------
        xi_t : Scalar. Current share of illiquid wealth.
        H_func : Function. Spline interpolator function for the H function.

        '''
        #optimize the Bellman equation given a xi_t
        # Minimize negative of (the log of the negative of) the value function (for optimization)
        objective = lambda params: self.ln_m_H_t_objective(xi_t, params) 
        #bounds = [(0, 1) for _ in range(self.mu_w.shape[0])] + [(0, .1)]  # Bounds for optimization
        #init_guess = np.append(0.2 * (1-xi_t)* np.ones(self.mu_w.shape[0]), 0.02*(1-xi_t))  # Initial guess for theta and c        
        result = minimize(objective, self.init_guess,method='Nelder-Mead') # bounds=bounds',L-BFGS-B,  
        theta_t_opt, c_t_opt = result.x[:-1], result.x[-1]
        #print(result)

        if not result.success: 
            print(f"Optimization convergence: {result.success}")
        #else: 
        #    print('No solution')
        
        ln_m_H_t_val_opt = result.fun  # Get the maximum value of the function (negative of objective)
        return ln_m_H_t_val_opt, theta_t_opt, c_t_opt
    
    def getH_str(self):
        '''
        get finer grid and evaluate for the optimum of H()
        '''
        ln_m_H_grid = self.ln_m_H_func(self.xi_fine_grid)
        #theta_fine_grid = self.theta_func(self.xi_fine_grid)
        c_fine_grid = self.c_func(self.xi_fine_grid)
        
        ln_m_H_star = min(ln_m_H_grid)
        
        str_index = np.argmin(ln_m_H_grid)
        
        xi_star = self.xi_fine_grid[str_index]
        #theta_star = theta_fine_grid[str_index]*(1-xi_star)
        c_star = c_fine_grid[str_index]*(1-xi_star)
        return ln_m_H_star, xi_star, c_star
    
    def fit_spline(self, y_value, fitSplit=True):
        if fitSplit == True: 
            # Split the data into two parts
            mask = self.Xi_t < 0.8
            x1, y1 = self.Xi_t[mask], y_value[mask]
            x2, y2 = self.Xi_t[~mask], y_value[~mask]
            
            # Fit the first spline
            spline1 = UnivariateSpline(x1, y1, k=2)
            
            # Fit the second spline
            spline2 = UnivariateSpline(x2, y2, k=2)
            
            # Combine the two splines into a single function
            def fit_fn(x):
                'combine the splines'
                if np.isscalar(x):
                    if x < 0.8:
                        return spline1(x)
                    else:
                        return spline2(x)
                else:
                    result = np.empty_like(x)
                    mask = x < 0.8
                    result[mask] = spline1(x[mask])
                    result[~mask] = spline2(x[~mask])
                    return result
        else: 
            fit_fn = UnivariateSpline(self.Xi_t, y_value, k=2) #, fill_value = 'extrapolate'

        return fit_fn

    def BellmanIterSolve(self, tol=1e-5, max_iter=500):
        # Store the optimal controls and value function
        self.theta_opt = np.zeros((self.gridpoints_Xi, len(self.mu_w)))
        self.c_opt = np.zeros(self.gridpoints_Xi)
    
        # Initialize with the Merton solutions
        self.ln_m_H_t_vals_opt_k = np.log(-np.ones_like(self.Xi_t)*self.H_m)
        #for jc, xi in enumerate(self.Xi_t):
        #    self.ln_m_H_t_vals_opt_k[jc] = self.H_m * np.ones_like(self.Xi_t) # self.utility(.03*(1-xi))
        self.ln_m_H_func = self.fit_spline(self.ln_m_H_t_vals_opt_k)
        self.ln_m_H_star = self.ln_m_H_t_vals_opt_k[0]
    
        # Enable interactive mode for live plotting
        #plt.ion()
        #axs = None  # Initialize axis object
        #lines = None  # Initialize line objects
        
        for k in range(max_iter):
            #print(f'Iteration {k}')
            self.init_guess = np.append(0.2*(1-.01)* np.ones(self.mu_w.shape[0]), 0.02*(1-.01))  # Initial guess for theta and c        
            for j, xi_j in enumerate(self.Xi_t):
                #print(f'xi_j = {xi_j:.2f}')
                self.ln_m_H_t_vals_opt_k[j], self.theta_opt[j, :], self.c_opt[j] = self.optimize(xi_j)
                self.init_guess = np.append(self.theta_opt[j, :], self.c_opt[j])  # Initial guess for theta and c
                #if self.H_t_vals_opt_k[j] > 0 :
                #    print('Positive value for xi_t : ', xi_j)
    
            # Compute the error between current and previous value functions
            error = np.linalg.norm(self.ln_m_H_func(self.Xi_t) - self.ln_m_H_t_vals_opt_k)
            # fit spline over ln -H(xi)
            self.ln_m_H_func = self.fit_spline(self.ln_m_H_t_vals_opt_k)
            #if any(self.H_func(self.xi_fine_grid)> 0):
            #    print('We have a problem')
            # Fit splines also on c_opt and theta_opt
            #TODO - fix this later for N> 1 liquid assets
            #self.theta_func = self.fit_spline(self.theta_opt[0].flatten())
            self.c_func = self.fit_spline(self.c_opt)
            'get H_str and xi_str'
            #self.H_star, self.xi_star, self.theta_star, self.c_star = self.getH_str()
            self.ln_m_H_star, self.xi_star, self.c_star = self.getH_str()


            # Print every k-th iteration
            if k % 25 == 0:
                print(f"Iteration {k}: ")
                print(f"               Value Fn Diff = {error:.6f}")
                print(f"               -ln(-H*) = {-self.ln_m_H_star:.4f}")
                print(f"               xi* = {self.xi_star:.4f}")
                #fig = plt.figure(figsize=(8, 6))
                #axs, lines = self.plot_results(axs, lines, iteration=k)
                #ax = fig.add_subplot(111)
                #ax.plot(self.Xi_t, -self.ln_m_H_t_vals_opt_k)
                #ax.plot(self.Xi_t, -self.ln_m_H_func(self.Xi_t))
    
            # Stop if the error is below the tolerance
            if error < tol:
                print(f"Converged in {k+1} iterations.")
                #evaluate Certainty Equivalents with final H_function function 
                self.cec_H_illiq = self.getCec(self.H_func(self.Xi_t))
                break
        else:
            print("Failed to converge within the maximum iterations.")
    
        # Keep the plot open after convergence
        #plt.ioff()
        #plt.show()

    def plot_results(self, axs=None, lines=None, iteration=None):
        """
        Update the value function, optimal consumption, and portfolio weights in the same plot.
        The behavior is equivalent to MATLAB's 'hold on'.
        """
        if axs is None:  # Initialize subplots only once
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # Set titles and labels for subplots
        axs[0].set_title("Value Function: log(-H_t_vals_opt_k)")
        axs[0].set_xlabel("xi_t")
        axs[0].set_ylabel("log(-H)")
        axs[0].plot(self.Xi_t, -self.ln_m_H_t_vals_opt_k)        
        
        axs[1].set_title("Optimal Consumption (c_opt)")
        axs[1].set_xlabel("xi_t")
        axs[1].set_ylabel("Consumption (c_t)")
        axs[1].plot(self.Xi_t, self.c_opt*(1-self.Xi_t))

        
        axs[2].set_title("Optimal Portfolio Weights (theta_opt)")
        axs[2].set_xlabel("xi_t")
        axs[2].set_ylabel("theta_t")
        axs[2].plot(self.Xi_t, self.theta_opt[:,0]*(1-self.Xi_t))
        plt.show()
        
        '''        
        # Redraw the figure to reflect updates
        plt.draw()
        plt.pause(0.01)  # Pause to ensure the plot is updated
        '''                
        return axs, lines  # Return updated axes and line objects

if __name__ == "__main__":
    # You can specify eta = 1 for one-year variables
    
    # Define parameters
    mu = np.array([0.055, 0.055])  # Example: two liquid assets and one illiquid asset
    Sigma = np.array([[0.14**2,0.], [0.,0.14**2]])
    gamma = 6.0
    beta = 0.03
    # illiquid asset can be solve once in 10 years on average
    eta = 1/10 
    r = 0.02
    dt = 1/20

    # Run 1 Year model
    model = IlliquidAssetModel(mu, Sigma, gamma, beta, eta, r, dt)
    model.solve()
    model.plot_results()
    
    plt.plot
