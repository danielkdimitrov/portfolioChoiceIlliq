# -*- coding: utf-8 -*-
"""
run_three_strategies.py

Reproduces three allocation strategies for a universe with two liquid risky assets
and one illiquid risky asset (last). Parameters follow the baseline in Table 1 of the
attached paper.

Strategies:
1) Merton (all assets liquid)
2) Heuristic Ang-style: (i) Merton within liquid assets only -> build a liquid block, (ii) solve illiquid 2-asset model (liquid block + illiquid)
3) Full illiquid model: optimize liquid and illiquid jointly (IlliquidAssetModel with 3 assets)

Outputs:
- Allocations (cash, liquid1, liquid2, illiquid) for each strategy
- xi* (optimal illiquid share), consumption at xi*, certainty equivalent consumption (CEC)
- Comparison highlighting differences between (2) and (3)
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# Import class from the provided module
from illiquidAssetModel import IlliquidAssetModel

np.set_printoptions(precision=5, suppress=True)

@dataclass
class Params:
    mu: np.ndarray        # (3,) risky asset expected returns (liquid1, liquid2, illiquid)
    sigma_vec: np.ndarray # (3,) volatilities
    rho_p2_priv: float    # correlation between Public 2 and Private
    r: float = 0.031
    beta: float = 0.031
    gamma: float = 6.0
    eta: float = 1/10     # average 10y between trades for the illiquid asset
    dt: float = 1.0
    illiq_premium: float = 0.0  # extra return on the illiquid asset


def build_sigma(sigma_vec: np.ndarray, rho_p2_priv: float) -> np.ndarray:
    """Construct 3x3 covariance matrix with Corr(P1,P2)=0, Corr(P1,Priv)=0, Corr(P2,Priv)=rho."""
    corr = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, rho_p2_priv],
        [0.0, rho_p2_priv, 1.0],
    ])
    Sigma = np.outer(sigma_vec, sigma_vec) * corr
    return Sigma


def merton_closed_form(mu: np.ndarray, Sigma: np.ndarray, r: float, gamma: float, beta: float) -> Tuple[np.ndarray, float, float]:
    """
    Closed-form Merton solution for continuous trading with CRRA utility.
    Returns (pi_opt, c_star, H_scalar) where pi are risky weights (w.r.t. total wealth),
    c_star is the optimal consumption rate per unit wealth, and H is the reduced value scalar.
    """
    # Avoid numerical issues
    Sigma_inv = np.linalg.pinv(Sigma)
    ones = np.ones(len(mu))
    mu_ex = mu - r * ones
    pi = (1.0 / gamma) * Sigma_inv @ mu_ex
    # Market price of risk norm
    lam2 = float(mu_ex.T @ Sigma_inv @ mu_ex)
    c_star = (beta + r * (gamma - 1)) / gamma + 0.5 * (gamma - 1) / (gamma ** 2) * lam2
    # Reduced value scalar H (see Annex B)
    H = (1.0 / (1.0 - gamma)) * (1.0 / c_star) ** gamma
    return pi, c_star, H


def liquid_block_from_two_assets(mu_L: np.ndarray, Sigma_LL: np.ndarray, r: float, gamma: float, beta: float) -> Tuple[float, float, np.ndarray]:
    """Build the liquid block using Merton within the two-liquid-only universe.
    Returns (mu_block, var_block, w_within_liquid) where w_within_liquid are weights inside block.
    """
    pi_L, _, _ = merton_closed_form(mu_L, Sigma_LL, r, gamma, beta)
    # Normalize to within-liquid composition
    w_within = pi_L / np.sum(pi_L) if np.sum(pi_L) != 0 else np.array([0.5, 0.5])
    mu_block = float(w_within @ mu_L)
    var_block = float(w_within @ Sigma_LL @ w_within)
    return mu_block, var_block, w_within


def run_strategies(p: Params) -> Dict[str, dict]:
    # Build 3-asset inputs
    mu = p.mu.copy()
    mu[-1] = mu[-1] + p.illiq_premium  # add illiquidity premium if any
    Sigma = build_sigma(p.sigma_vec, p.rho_p2_priv)

    # 1) Merton with all assets liquid
    pi_3m, c_3m, H_3m = merton_closed_form(mu, Sigma, p.r, p.gamma, p.beta)
    alloc_3m = np.hstack([1 - np.sum(pi_3m), pi_3m])

    # 2) Heuristic Ang-style two-step
    # Step 1: Merton within the two liquid assets only
    mu_L = mu[:-1]
    Sigma_LL = Sigma[:-1, :-1]
    mu_block, var_block, w_within = liquid_block_from_two_assets(mu_L, Sigma_LL, p.r, p.gamma, p.beta)
    # Correlation / covariance between block and illiquid
    cov_LX_vec = Sigma[:-1, -1]
    cov_block_X = float(w_within @ cov_LX_vec)
    var_X = float(Sigma[-1, -1])
    # Build 2-asset model (liquid block + illiquid)
    mu_2 = np.array([mu_block, mu[-1]])
    Sigma_2 = np.array([[var_block, cov_block_X], [cov_block_X, var_X]])

    model_heur = IlliquidAssetModel(mu_2, Sigma_2, p.gamma, p.beta, p.eta, p.r, p.dt, True)
    model_heur.BellmanIterSolve(tol=1e-4, max_iter=400)
    model_heur.getFinalResults()
    # Unpack allocations: [cash, liquid_block, illiquid]
    cash_block_ill = model_heur.alloc
    liquid_block_weight = cash_block_ill[1]
    illiq_weight = cash_block_ill[2]
    # Map back to original assets
    liquid1 = liquid_block_weight * w_within[0]
    liquid2 = liquid_block_weight * w_within[1]
    cash_heur = cash_block_ill[0]
    alloc_heur = np.array([cash_heur, liquid1, liquid2, illiq_weight])

    # 3) Full illiquid model with three assets (two liquid risky + one illiquid risky)
    model_full = IlliquidAssetModel(mu, Sigma, p.gamma, p.beta, p.eta, p.r, p.dt, True)
    model_full.BellmanIterSolve(tol=1e-4, max_iter=400)
    model_full.getFinalResults()
    # model_full.alloc = [cash, liquid1, liquid2, illiquid]
    alloc_full = model_full.alloc

    # Build results dictionary
    res = {
        "merton_all_liquid": {
            "alloc": alloc_3m,
            "c_rate": c_3m,
            "H": H_3m,
        },
        "heuristic_two_step": {
            "alloc": alloc_heur,
            "xi_star": model_heur.xi_star,
            "c_star_xi": model_heur.c_star_xi,
            "CEC_star": model_heur.cec_il_star,
            "CEC_liquid_only": model_heur.cec_m2,
            "allocation_benefit_star": model_heur.allocationBenefit_star,
        },
        "full_illiquid_joint": {
            "alloc": alloc_full,
            "xi_star": model_full.xi_star,
            "c_star_xi": model_full.c_star_xi,
            "CEC_star": model_full.cec_il_star,
            "CEC_liquid_only": model_full.cec_m2,
            "allocation_benefit_star": model_full.allocationBenefit_star,
        },
        "meta": {
            "rho_p2_priv": p.rho_p2_priv,
            "illiq_premium": p.illiq_premium,
            "params": p,
            "within_liquid_weights_for_block": w_within,
        }
    }
    return res


def pretty_percent(x):
    return f"{100*x:6.2f}%"


def print_summary(res: Dict[str, dict]):
    print("=== Strategy Comparison (weights are fractions of total wealth) ===")
    labels = ["Cash", "Public1", "Public2", "Private (illiquid)"]

    def row(alloc, title):
        print(f"\n{title}")
        for lab, w in zip(labels, alloc):
            print(f"  {lab:18s}: {pretty_percent(w)}")
        print(f"  Sum            : {np.sum(alloc):.4f}")

    row(res["merton_all_liquid"]["alloc"], "(1) Merton (all assets liquid)")
    row(res["heuristic_two_step"]["alloc"], "(2) Heuristic Ang-style (liquid block + illiquid)")
    row(res["full_illiquid_joint"]["alloc"], "(3) Full model with illiquidity (joint optimization)")

    print("\n--- Key Scalars ---")
    print(f"rho(Public2, Private) = {res['meta']['rho_p2_priv']:.2f}  | illiquidity premium (private) = {res['meta']['illiq_premium']:.2%}")

    # xi* and CECs
    print("\nHeuristic two-step:")
    print(f"  xi* (illiquid share)  : {res['heuristic_two_step']['xi_star']:.4f}")
    print(f"  c*(xi*)               : {res['heuristic_two_step']['c_star_xi']:.4f}")
    print(f"  CEC*(illiquid)        : {res['heuristic_two_step']['CEC_star']:.4f}")
    print(f"  CEC (liquid-only)     : {res['heuristic_two_step']['CEC_liquid_only']:.4f}")

    print("\nFull model (joint):")
    print(f"  xi* (illiquid share)  : {res['full_illiquid_joint']['xi_star']:.4f}")
    print(f"  c*(xi*)               : {res['full_illiquid_joint']['c_star_xi']:.4f}")
    print(f"  CEC*(illiquid)        : {res['full_illiquid_joint']['CEC_star']:.4f}")
    print(f"  CEC (liquid-only)     : {res['full_illiquid_joint']['CEC_liquid_only']:.4f}")

    # Difference between heuristic and full allocations (L1 norm and component diffs)
    a_h = res['heuristic_two_step']['alloc']
    a_f = res['full_illiquid_joint']['alloc']
    print("\nAllocation differences (Full - Heuristic):")
    for lab, dh in zip(["Cash", "Public1", "Public2", "Private"], a_f - a_h):
        print(f"  {lab:10s}: {dh:+.4f}")
    print(f"  L1 distance: {np.sum(np.abs(a_f - a_h)):.4f}")


if __name__ == "__main__":
    # Baseline parameters: Table 1
    mu = np.array([0.067, 0.067, 0.067])
    sigma = np.array([0.1626, 0.1626, 0.1626])

    # You can toggle rho and illiquidity premium here
    scenarios = [
        (0.00, 0.00),  # high diversification, no premium
        (0.80, 0.00),  # low diversification, no premium
        # (0.00, 0.03),  # high diversification, +3% premium (uncomment to test)
        # (0.80, 0.03),  # low diversification, +3% premium (uncomment to test)
    ]

    for rho, prem in scenarios:
        print("\n==============================================================")
        print(f"Running scenario: rho(P2,Priv)={rho:.2f}, illiq premium={prem:.2%}")
        p = Params(mu=mu, sigma_vec=sigma, rho_p2_priv=rho, illiq_premium=prem)
        res = run_strategies(p)
        print_summary(res)
