
import numpy as np
from IlliquidAssetModelTwoAssets import IlliquidAssetModelTwoAssets

params = {
    "gamma": 5.0,
    "r": 0.02,
    "mu_w": np.array([0.06, 0.07]),
    "sigma_w": np.array([[0.15, 0.02], [0.01, 0.2]]),
    "mu_x": 0.08,
    "mu_y": 0.09,
    "sigma_x": np.array([0.1, 0.1]),
    "sigma_y": np.array([0.05, 0.2]),
    "eta1": 1.0,
    "eta2": 0.5,
    "delta": 0.96,
    "dt": 1/12,
    "Nq": 3,
    "gridpoints": 15
}

model = IlliquidAssetModelTwoAssets(params)
model.BellmanIterSolve(max_iter=100, tol=1e-5)
