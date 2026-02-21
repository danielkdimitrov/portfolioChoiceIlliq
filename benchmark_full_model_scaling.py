# -*- coding: utf-8 -*-
"""
benchmark_full_model_scaling.py

Purpose
-------
Benchmark the performance of the full illiquid portfolio choice model as the number
of LIQUID risky assets increases from 1 to 10 (plus one illiquid risky asset).
For each dimension, solve the full model twice:
  (i) using Gauss–Hermite quadrature (useQuadrature=True), and
  (ii) using Monte Carlo simulation (useQuadrature=False).

For each case, record wall-clock runtime (seconds), extract the optimal strategic
allocation (cash + each liquid + illiquid), and compute the L1 distance between the
quadrature and Monte Carlo allocations. Produce:
  - A printed summary table of runtimes and L1 differences.
  - Two plots saved to disk:
        1) runtime_vs_assets.png    (runtime vs. # of liquid assets)
        2) alloc_L1_diff_vs_assets.png (L1 difference vs. # of liquid assets)

Notes
-----
- This script relies on the IlliquidAssetModel class provided in illiquidAssetModel.py.
- Quadrature node count grows exponentially in the number of risk factors (m^n with m=4),
  so high-dimensional runs (>= 8 liquids) may take significant time and memory.
- Monte Carlo inside the class currently uses a fixed number of simulations (nSims=5,000).

Author: Copilot (based on Dimitrov (2026) model code)
"""
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from illiquidAssetModel import IlliquidAssetModel

# ----------------------
# Global configuration
# ----------------------
GAMMA = 6.0
BETA = 0.031
R_RF = 0.031
ETA = 1/10          # average 10 years between trades
DT = 1.0
TOL = 1e-4
MAX_ITER = 400

MU_RISKY = 0.067
SIGMA_RISKY = 0.1626

# Correlation between the last liquid asset and the illiquid asset, to keep a
# non-trivial hedging channel. Other liquid-liquid correlations are set to zero.
RHO_LIQ_LAST_WITH_ILL = 0.80

# Reproducibility for Monte Carlo path draws inside the model (where applicable).
np.random.seed(12345)


def build_inputs(n_liquid: int):
    """Construct (mu, Sigma) for a universe with n_liquid liquid risky assets
    and 1 illiquid risky asset (last index). Liquids are mutually uncorrelated; only the
    LAST liquid is correlated with the illiquid asset at RHO_LIQ_LAST_WITH_ILL.
    """
    n_total = n_liquid + 1  # +1 for illiquid
    mu = np.full(n_total, MU_RISKY, dtype=float)
    sig = np.full(n_total, SIGMA_RISKY, dtype=float)

    corr = np.eye(n_total)
    if n_liquid >= 1:
        # correlate last liquid (index n_liquid-1) with illiquid (index n_liquid)
        i_last_liq = n_liquid - 1
        i_illiq = n_liquid
        corr[i_last_liq, i_illiq] = RHO_LIQ_LAST_WITH_ILL
        corr[i_illiq, i_last_liq] = RHO_LIQ_LAST_WITH_ILL
    Sigma = np.outer(sig, sig) * corr
    return mu, Sigma


def run_model(mu, Sigma, use_quadrature: bool):
    """Instantiate and solve the full model under the requested expectation engine."""
    model = IlliquidAssetModel(mu, Sigma, GAMMA, BETA, ETA, R_RF, DT, useQuadrature=use_quadrature)
    t0 = time.perf_counter()
    model.BellmanIterSolve(tol=TOL, max_iter=MAX_ITER)
    model.getFinalResults()
    t1 = time.perf_counter()
    runtime = t1 - t0
    return runtime, model.alloc, model


def main():
    dims = list(range(1, 11))  # 1..10 liquid risky assets
    times_quad = []
    times_mc = []
    l1_diffs = []

    print("Benchmarking full model from 1 to 10 liquid risky assets (plus 1 illiquid)\n")
    print(f"Settings: gamma={GAMMA}, r=beta={R_RF:.3f}, eta=1/10, dt={DT}, tol={TOL}, max_iter={MAX_ITER}")
    print(f"mu={MU_RISKY:.3f}, sigma={SIGMA_RISKY:.4f}; Corr(liq_last,ill)={RHO_LIQ_LAST_WITH_ILL:.2f}\n")

    for nL in dims:
        print(f"--- n_liquid = {nL} (total risky = {nL+1}) ---")
        mu, Sigma = build_inputs(nL)

        # Quadrature run
        alloc_q = None
        try:
            rt_q, alloc_q, _ = run_model(mu, Sigma, use_quadrature=True)
            times_quad.append(rt_q)
            print(f"  Quadrature: {rt_q:8.2f} s | alloc length = {len(alloc_q)}")
        except Exception as e:
            warnings.warn(f"Quadrature failed for nL={nL}: {e}")
            times_quad.append(np.nan)
            print(f"  Quadrature: FAILED ({e})")

        # Monte Carlo run
        alloc_m = None
        try:
            rt_m, alloc_m, _ = run_model(mu, Sigma, use_quadrature=False)
            times_mc.append(rt_m)
            print(f"  Monte Carlo: {rt_m:8.2f} s | alloc length = {len(alloc_m)}")
        except Exception as e:
            warnings.warn(f"Monte Carlo failed for nL={nL}: {e}")
            times_mc.append(np.nan)
            print(f"  Monte Carlo: FAILED ({e})")

        # L1 difference if both succeeded and dimensions match
        if alloc_q is not None and alloc_m is not None and len(alloc_q) == len(alloc_m):
            l1 = float(np.sum(np.abs(alloc_q - alloc_m)))
        else:
            l1 = np.nan
        l1_diffs.append(l1)
        print(f"  L1(alloc_full_quadrature - alloc_full_mc) = {l1 if np.isfinite(l1) else np.nan}\n")

    # Summary print
    print("\n=== Summary ===")
    print("n_liquid | time_quad[s] | time_mc[s] | L1 alloc diff")
    for nL, tq, tm, l1 in zip(dims, times_quad, times_mc, l1_diffs):
        print(f"{nL:8d} | {tq:12.3f} | {tm:10.3f} | {l1}")

    # Plot: runtime vs dimension
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(dims, times_quad, 'o-', label='Quadrature')
    ax.plot(dims, times_mc, 's--', label='Monte Carlo')
    ax.set_xlabel('# liquid risky assets')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime vs. number of liquid risky assets')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('runtime_vs_assets.pdf', dpi=160)

    # Plot: L1 allocation difference
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.0))
    ax2.plot(dims, l1_diffs, 'd-', color='tab:red')
    ax2.set_xlabel('# liquid risky assets')
    ax2.set_ylabel('L1 difference in optimal allocations')
    ax2.set_title('Quadrature vs. Monte Carlo: L1(alloc)')
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig('alloc_L1_diff_vs_assets.pdf', dpi=160)

    print("\nSaved plots: runtime_vs_assets.pdf, alloc_L1_diff_vs_assets.pdf")


if __name__ == '__main__':
    main()
