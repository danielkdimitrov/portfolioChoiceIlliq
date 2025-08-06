
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss


class IlliquidAssetModelTwoAssets:
    def __init__(self, params):
        self.gamma = params["gamma"]
        self.r = params["r"]
        self.mu_w = params["mu_w"]
        self.sigma_w = params["sigma_w"]
        self.mu_x = params["mu_x"]
        self.mu_y = params["mu_y"]
        self.sigma_x = params["sigma_x"]
        self.sigma_y = params["sigma_y"]
        self.eta1 = params["eta1"]
        self.eta2 = params["eta2"]
        self.delta = params["delta"]
        self.dt = params["dt"]
        self.Nq = params["Nq"]
        self.gridpoints = params["gridpoints"]

        self.n_shocks = self.sigma_w.shape[0]
        self.build_state_grid()
        self.build_quadrature()

        self.H = np.ones(self.state_points.shape[0]) * 0.1
        self.update_interpolator()

    def build_state_grid(self):
        lin = np.linspace(0, 0.99, self.gridpoints)
        Xi_X, Xi_Y = np.meshgrid(lin, lin)
        valid = Xi_X + Xi_Y < 0.99
        self.Xi_X = Xi_X[valid]
        self.Xi_Y = Xi_Y[valid]
        self.state_points = np.column_stack((self.Xi_X, self.Xi_Y))

    def build_quadrature(self):
        z, w = hermgauss(self.Nq)
        z *= np.sqrt(2)
        w /= np.sqrt(np.pi)
        self.Z = np.array(np.meshgrid(*([z] * self.n_shocks))).T.reshape(-1, self.n_shocks)
        self.W = np.prod(np.array(np.meshgrid(*([w] * self.n_shocks))).T.reshape(-1, self.n_shocks), axis=1)

    def update_interpolator(self):
        self.H_interp = LinearNDInterpolator(self.state_points, np.log(self.H))

    def u(self, c):
        return c ** (1 - self.gamma) / (1 - self.gamma)

    def wealth_growth(self, xi_x, xi_y, theta, c, dZ):
        R_w = 1 + (self.r + theta @ (self.mu_w - self.r)) * self.dt + theta @ self.sigma_w @ dZ * np.sqrt(self.dt)
        R_x = 1 + self.mu_x * self.dt + self.sigma_x @ dZ * np.sqrt(self.dt)
        R_y = 1 + self.mu_y * self.dt + self.sigma_y @ dZ * np.sqrt(self.dt)
        R_q = (1 - xi_x - xi_y) * R_w + xi_x * R_x + xi_y * R_y
        xi_x_next = xi_x * R_x / R_q
        xi_y_next = xi_y * R_y / R_q
        return R_q, xi_x_next, xi_y_next

    def expected_H(self, xi_x, xi_y, theta, c):
        result = 0.0
        for z, w in zip(self.Z, self.W):
            R_q, xi_x_next, xi_y_next = self.wealth_growth(xi_x, xi_y, theta, c, z)
            if xi_x_next + xi_y_next >= 0.99:
                continue
            H_next = self.H_interp(xi_x_next, xi_y_next)
            if np.isnan(H_next):
                continue
            result += w * np.exp(H_next) * R_q ** (1 - self.gamma)
        return result

    def H_objective(self, xi_x, xi_y, p):
        def obj(pc):
            c = pc[0]
            theta = pc[1:]
            if c <= 0 or c >= 1 or np.any(np.abs(theta) > 10):
                return 1e6
            u_val = self.u(c * (1 - xi_x - xi_y)) * self.dt
            exp_val = self.expected_H(xi_x, xi_y, theta, c)
            rebalance_X = np.max([np.exp(self.H_interp(x, xi_y)) for x in np.linspace(0, 0.99 - xi_y, 20)])
            rebalance_Y = np.max([np.exp(self.H_interp(xi_x, y)) for y in np.linspace(0, 0.99 - xi_x, 20)])
            cont = (1 - p[0] - p[1]) * exp_val
            jump_x = p[0] * rebalance_X
            jump_y = p[1] * rebalance_Y
            return - (u_val + self.delta * (cont + jump_x + jump_y))

        return obj

    def optimize_at_point(self, xi_x, xi_y):
        p1 = 1 - np.exp(-self.eta1 * self.dt)
        p2 = 1 - np.exp(-self.eta2 * self.dt)
        p = (p1, p2)
        guess = np.array([0.05] + [0.0] * self.n_shocks)
        res = minimize(self.H_objective(xi_x, xi_y, p), guess, method='Nelder-Mead')
        return -res.fun if res.success else -1e6

    def BellmanIterSolve(self, max_iter=500, tol=1e-6):
        for k in range(max_iter):
            H_new = np.zeros_like(self.H)
            for i, (xi_x, xi_y) in enumerate(self.state_points):
                H_new[i] = self.optimize_at_point(xi_x, xi_y)
            diff = np.linalg.norm(np.log(-H_new + 1e-8) - np.log(-self.H + 1e-8))
            self.H = H_new
            self.update_interpolator()
            if diff < tol:
                print(f"Converged at iteration {k}")
                break
