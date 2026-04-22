"""
scaling_fits.py — Extract c from Ω(N) data
==========================================
Fits Ω(N) = 1/(1 + A·N^c) to extract the scaling exponent c
for each temperature (SYK) or horizon configuration (BH sonic).

Also computes ΔAIC between algebraic (logistic) and power-law fits.

Usage: python scaling_fits.py
"""
import numpy as np
from scipy.optimize import curve_fit
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def algebraic(N, A, c):
    """Ω = 1/(1 + A·N^c) — logistic saturation (Hip. A)."""
    return 1.0 / (1.0 + A * N ** c)

def power_law(N, alpha, gamma):
    """Ω = alpha · N^(-gamma) — power-law decay (Hip. C)."""
    return alpha * N ** (-gamma)

def fit_and_compare(N_values, omega_values, sigma=None, label=""):
    """Fit both models, compute ΔAIC, return results."""
    N = np.array(N_values, dtype=float)
    Om = np.array(omega_values)
    
    results = {"N": list(N_values), "Omega": list(omega_values)}
    
    # Algebraic fit
    try:
        if sigma is not None:
            popt_a, pcov_a = curve_fit(algebraic, N, Om, sigma=sigma,
                                       p0=[0.1, 2.0], maxfev=50000)
        else:
            popt_a, pcov_a = curve_fit(algebraic, N, Om,
                                       p0=[0.1, 2.0], maxfev=50000)
        
        A, c = popt_a
        c_err = np.sqrt(pcov_a[1, 1]) if pcov_a[1, 1] > 0 else 0
        Om_pred = algebraic(N, *popt_a)
        ss_res = np.sum((Om - Om_pred) ** 2)
        ss_tot = np.sum((Om - np.mean(Om)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        results['algebraic'] = {
            'A': float(A), 'c': float(c), 'c_err': float(c_err),
            'R2': float(R2), 'RSS': float(ss_res)
        }
    except Exception as e:
        results['algebraic'] = {'error': str(e)}
    
    # Power-law fit
    try:
        popt_p, pcov_p = curve_fit(power_law, N, Om, p0=[1.0, 1.0], maxfev=50000)
        Om_pred_p = power_law(N, *popt_p)
        ss_res_p = np.sum((Om - Om_pred_p) ** 2)
        R2_p = 1 - ss_res_p / ss_tot if ss_tot > 0 else 0
        
        results['power_law'] = {
            'alpha': float(popt_p[0]), 'gamma': float(popt_p[1]),
            'R2': float(R2_p), 'RSS': float(ss_res_p)
        }
    except Exception as e:
        results['power_law'] = {'error': str(e)}
    
    # ΔAIC (both have k=2 parameters)
    if 'RSS' in results.get('algebraic', {}) and 'RSS' in results.get('power_law', {}):
        n = len(N)
        rss_a = results['algebraic']['RSS']
        rss_p = results['power_law']['RSS']
        if rss_a > 0 and rss_p > 0:
            daic = n * np.log(rss_p / rss_a)
            results['DAIC'] = float(daic)
    
    if label:
        print(f"  {label}: c={results.get('algebraic', {}).get('c', '?'):.4f}, "
              f"R²={results.get('algebraic', {}).get('R2', '?'):.4f}, "
              f"ΔAIC={results.get('DAIC', '?')}")
    
    return results

if __name__ == "__main__":
    print("scaling_fits.py — run verify_reproduction.py for full analysis")
