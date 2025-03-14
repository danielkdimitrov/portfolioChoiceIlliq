import numpy as np
from numpy.polynomial.hermite import hermgauss
import itertools
from scipy.linalg import cholesky
from scipy.interpolate import UnivariateSpline, CubicSpline, interp1d, InterpolatedUnivariateSpline
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt  


class IlliquidAssetModel:
    def __init__(self, mu, Sigma, gamma, beta, eta, r, dt=1, useQuadrature=True, longOnly = True):
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
        self.m = 4 #number of quadrature points -> 4
        
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
        self.pi_m, self.c_m, self.H_m = self.merton_solution(self.mu, self.sigma)
        self.alloc_m = np.hstack([1-sum(self.pi_m), self.pi_m])
        self.cec_m = self.getCec(self.H_m)

        # Merton liquid assets only
        self.pi_m2, self.c_m2, self.H_m2 = self.merton_solution(self.mu_w, self.sigma_w[:,:-1])
        self.alloc_2m = np.hstack([1-sum(self.pi_m2), self.pi_m2])
        self.cec_m2 = self.getCec(self.H_m2)
        
        # Generate quadrature points and weights
        self.useQuadrature = useQuadrature
        if self.useQuadrature == True:
            'Quadrate model'
            xn, self.Wn = self.generate_quadrature_points()
            self.dZ = xn*np.sqrt(2)
        else:
            'Simulated model'
            nSims = 5*10**3
            self.dZ = self.generate_standard_normal(nSims, self.n)
        
        self.longOnly = longOnly 
        # Create a grid over xi
        self.gridpoints_Xi = 20
        self.Xi_t = np.linspace(0.,.99, self.gridpoints_Xi)
        
        # Finer grid 
        self.xi_fine_grid = np.linspace(.0, .99, 2000)
        
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
        
        
    def merton_solution(self, mu, sigma):
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
        mu_minus_r = mu - self.r * np.ones(len(mu))
        lambda_ = np.linalg.solve(sigma, mu_minus_r)
        
        # Optimal portfolio allocation: pi = (1 / gamma) * Sigma^{-1} * (mu - r1)
        pi_opt = (1 / self.gamma) * np.linalg.inv(sigma.T) @ lambda_
                
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
        #if c <= 0:
        #    return -np.inf #0.1**8  # Penalize zero or negative consumption
        if self.gamma == 1:
            return np.log(c)
        else:
            return c**(1 - self.gamma) / (1 - self.gamma)
        
    def generate_standard_normal(self, n, d):
        """
        Generates `n` samples of `d`-dimensional standard normal variables.
        
        Parameters:
        - n: int, the number of samples to generate
        - d: int, the number of dimensions for each sample
    
        Returns:
        - samples: numpy.ndarray, shape (n, d), each row is a d-dimensional standard normal vector
        """
        return np.random.randn(n, d)        
        
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
        # Liquid asset growth
        R_w = 1 + (self.r + theta_t @ (self.mu_w - self.r) - c_t) * self.dt + (theta_t @ self.sigma_w) @ self.dZ.T * np.sqrt(self.dt)
        
        # Illiquid asset growth
        R_x = 1 + self.mu_x * self.dt + self.sigma_x @ self.dZ.T * np.sqrt(self.dt)
        
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
        if self.useQuadrature == True:
            value = self.const*np.sum(self.Wn * transformed_nodes)
        else: 
            value = np.mean(transformed_nodes)
        return value

    
    def ln_m_H_t_objective(self, xi_t, params):
        theta_t, c_t = params[:-1], params[-1]
        #get next period's dynamic
        R_q, xi_next = self.wealth_growth(theta_t, c_t, xi_t)
        'Calculate the terms in the Bellman equation'
        if np.any(xi_next < 0) or np.any(xi_next > 1):  # This checks if any xi_next exceeds 1
            ln_m_H_t_val = 50 #2*self.ln_m_H_star    # or just fix to 50 
            #H_next_illiq = -np.inf  #np.inf  # Enforce H_func going to negative infinity when xi_next > 1            
            #c_t = 0.1**10
        #else:                 
        # get intermediate utility
        util = self.utility(c_t * (1 - xi_t))         
        # transform back the H_vals function from log minus
        H_vals_next = - np.exp(self.ln_m_H_func(xi_next))
        # get terms in case of illiquidity
        H_next_illiq = self.expectation(R_q**(1 - self.gamma) * H_vals_next)          
                        
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
        #optimize the Bellman equation given a xi_t; Minimize negative of (the log of the negative of) the value function (for optimization)
        objective = lambda params: self.ln_m_H_t_objective(xi_t, params) 
        bounds = [(0.001, 1) for _ in range(self.mu_w.shape[0])] + [(0.001, .1)]  # Bounds for optimization
        #init_guess = np.append(0.2 * (1-xi_t)* np.ones(self.mu_w.shape[0]), 0.02*(1-xi_t))  # Initial guess for theta and c        
        if self.longOnly == True:
            result = minimize(objective, self.init_guess[self.j_gridPoint ,:], bounds=bounds, method='L-BFGS-B',options={'maxiter': 1000, 'ftol': 1e-9}) # ',,   method = 'Nelder-Mead'
        else: 
            result = minimize(objective, self.init_guess[self.j_gridPoint ,:], method='Nelder-Mead') # ',,   method = 'Nelder-Mead'
        theta_t_opt, c_t_opt = result.x[:-1], result.x[-1]
        #print(result)

        if not result.success: 
            print(f"Optimization convergence: {result.success}")
        
        ln_m_H_t_val_opt = result.fun  # Get the maximum value of the function (negative of objective)
        return ln_m_H_t_val_opt, theta_t_opt, c_t_opt
    
    def getH_str(self, finalRun=False):
        '''
        get finer grid and evaluate for the optimum of H()
        '''
        ln_m_H_grid = self.ln_m_H_func(self.xi_fine_grid)
        ln_m_H_star = min(ln_m_H_grid)
        str_index = np.argmin(ln_m_H_grid)
        xi_star = self.xi_fine_grid[str_index]
        if finalRun == False:
            return ln_m_H_star, xi_star            
        else: 
            # final run calculations 
            'get c star'
            self.c_func = self.fit_spline(self.c_opt )
            #c_fine_grid = self.c_func(self.xi_fine_grid)
            # get c at xi_star and revaluate from total wealth
            c_star_xi = self.c_func(xi_star)*(1-xi_star) #c_fine_grid[str_index]*(1-xi_star)
            'get theta_star' 
            theta_star_xi = np.zeros(self.n-1)
            #collect thetas in a list
            self.theta_func =[]
            for j in range(self.n-1):
                theta_func = self.fit_spline(self.theta_opt[:,j])
                self.theta_func.append(theta_func)
                # get theta at xi_star and adjust for xi_star
                #theta_fine_grid = theta_func(self.xi_fine_grid)   
                theta_star_xi[j] =  theta_func(xi_star)*(1-xi_star)
            risky_alloc = np.hstack([theta_star_xi.T, xi_star])
            alloc = np.hstack([1- sum(risky_alloc), risky_alloc])
            return xi_star, c_star_xi, theta_star_xi, alloc
            #return ln_m_H_star, xi_star, theta_star_xi, c_star_xi
    
    def fit_spline(self, y_value, fitSplitSpline=False):
        if fitSplitSpline == False: 
            fit_fn = interp1d(self.Xi_t, y_value, kind='cubic', fill_value='extrapolate',bounds_error=False)
            #UnivariateSpline(self.Xi_t, y_value, k=2) #, ext=3, fill_value = 'extrapolate'
        else: 
            # Split the data into two parts
            mask = self.Xi_t < 0.8
            x1, y1 = self.Xi_t[mask], y_value[mask]
            x2, y2 = self.Xi_t[~mask], y_value[~mask]
            
            # Fit the first spline
            spline1 = UnivariateSpline(x1, y1, k=3)
            
            # Fit the second spline
            spline2 = UnivariateSpline(x2, y2, k=3)#, ext = 3
            
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
        return fit_fn

    def BellmanIterSolve(self, tol=1e-5, max_iter=900):
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
        self.init_guess = np.ones((len(self.Xi_t), len(self.mu_w)+1)) * np.append(0.1*np.ones_like(self.mu_w), .02)  # Initial guess for theta and c        
        
        
        for k in range(max_iter):
            #print(f'Iteration {k}')
            for j, xi_j in enumerate(self.Xi_t):
                self.j_gridPoint = j 
                #print(f'xi_j = {xi_j:.2f}')
                self.ln_m_H_t_vals_opt_k[j], self.theta_opt[j, :], self.c_opt[j] = self.optimize(xi_j)
                self.init_guess[j,:] = np.append(self.theta_opt[j, :], self.c_opt[j])  # Initial guess for theta and c
                #if self.H_t_vals_opt_k[j] > 0 :
                #    print('Positive value for xi_t : ', xi_j)
    
            # Compute the error between current and previous value functions
            error = np.linalg.norm(self.ln_m_H_func(self.Xi_t) - self.ln_m_H_t_vals_opt_k)
            # fit spline over ln -H(xi)
            self.ln_m_H_func = self.fit_spline(self.ln_m_H_t_vals_opt_k)
            #if any(self.H_func(self.xi_fine_grid)> 0):
            #    print('We have a problem')
            'get H_str and xi_str'
            self.ln_m_H_star, self.xi_star  = self.getH_str()


            # Print every k-th iteration
            if k % 100 == 0:
                print(f"Iteration {k}: ")
                print(f"               Value Fn Diff = {error:.6f}")
                print(f"               -ln(-H*) = {-self.ln_m_H_star:.4f}")
                print(f"               xi* = {self.xi_star:.4f}")
                #self.plot_results()
                #ax = fig.add_subplot(111)
                #ax.plot(self.Xi_t, -self.ln_m_H_t_vals_opt_k)
                #ax.plot(self.Xi_t, -self.ln_m_H_func(self.Xi_t))
    
            # Stop if the error is below the tolerance
            if error < tol:
                print(f"Converged in {k+1} iterations.")
                self.convergence = True
                break            
        if k == max_iter: 
            'In case of no convergence'
            self.convergence = False
            print("Failed to converge within the maximum iterations.")


        self.getFinalResults()
    
        # Keep the plot open after convergence
        #plt.ioff()
        #plt.show()
        
    def getFinalResults(self):
        'collect and save the final results'
        self.xi_star, self.c_star_xi, self.theta_star_xi, self.alloc  = self.getH_str(True)
        #evaluate Certainty Equivalents with final H_function function 
        self.cec_il = self.getCec(-np.exp(self.ln_m_H_func(self.Xi_t)))
        self.allocationBenefit = self.cec_il / self.cec_m2 - 1
        # Evaluate cec illiquid at xi_star
        self.cec_il_star = self.getCec(-np.exp(self.ln_m_H_func(self.xi_star)))
        self.allocationBenefit_star = self.cec_il_star / self.cec_m2 - 1
        # save convergence indicator
        # get simulated values (scaled for totl wealth)
        self.xi_sim, self.c_sim, self.theta_sim, self.transfer_sim = self.simulation()

    def simulation(self):
        # Initialize the path
        num_simulations = 2*10**4
        dt_sqrt = np.sqrt(self.dt)
        
        # Generate a 1000xN matrix of standard normal shocks
        dZ_path = np.random.multivariate_normal(mean=np.zeros(self.sigma.shape[0]), cov=np.eye(self.sigma.shape[0]), size=num_simulations)
        tradeIndic = np.random.binomial(1, self.p, num_simulations)
        
        xi_t = np.zeros(num_simulations)
        c_t = np.zeros(num_simulations)
        theta_t = np.zeros((num_simulations, self.n-1))
        transfer = np.zeros(num_simulations)
        # Initialize variables
        xi_t[0] = self.xi_star  # Set an initial value for xi_t
        X, W = np.zeros(num_simulations), np.zeros(num_simulations)
        W[0], X[0] = 1*(1-xi_t[0]), 1*(xi_t[0])
        
        for t in range(num_simulations-1):    
            # Calculate wealth growth
            for j in range(self.n-1):
                theta_t[t, j] = self.theta_func[j](xi_t[t])
            
            c_t[t] = self.c_func(xi_t[t])
            W[t+1] = W[t]* (1 + (self.r + theta_t[t,:]@ (self.mu_w - self.r) - c_t[t]) * self.dt + (theta_t[t,:] @ self.sigma_w @ dZ_path[t]) * dt_sqrt ) 
            
            # Calculate illiquid asset growth
            X[t+1] = X[t]* ( 1 + self.mu_x * self.dt + (self.sigma_x @ dZ_path[t]) * dt_sqrt)
            # Make transfers
            xi_before = X[t+1] / (W[t+1] + X[t+1])
            transfer[t] = (W[t+1]+X[t+1])*(self.xi_star - xi_before) * tradeIndic[t]
            W[t+1] = W[t+1]- transfer[t]
            X[t+1]= X[t+1] + transfer[t]
                                
            # Update xi_t for next iteration
            xi_t[t+1] = X[t+1] / (W[t+1]+ X[t+1])
        
        return xi_t, c_t*(1-xi_t), theta_t*(1-xi_t.reshape(-1, 1)), transfer
        
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
        axs[2].plot(self.Xi_t, self.theta_opt[:,1]*(1-self.Xi_t))
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
