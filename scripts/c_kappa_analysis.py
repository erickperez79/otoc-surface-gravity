"""
c_kappa_analysis.py — Fit c(κ) for SYK and BH sonic
====================================================
Fits c = a·ln(κ) + b (logarithmic) vs c = a/κ + b (inverse)
for both models. Computes ΔAIC and R² for model comparison.

Usage: python c_kappa_analysis.py [--plot]
"""
import numpy as np
from scipy.optimize import curve_fit
import json
import os
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def log_model(kappa, a, b):
    return a * np.log(kappa) + b

def inv_model(kappa, a, b):
    return a / kappa + b

def lin_model(kappa, a, b):
    return a * kappa + b

def fit_c_kappa(kappas, c_vals, c_errs=None, label=""):
    """Fit log, inv, and lin models. Return best fit and ΔAIC."""
    kappa = np.array(kappas)
    c = np.array(c_vals)
    sigma = np.array(c_errs) if c_errs is not None else None
    
    results = {}
    
    for name, model, p0 in [('log', log_model, [0.5, 0.5]),
                             ('inv', inv_model, [1.0, 1.0]),
                             ('lin', lin_model, [0.1, 1.0])]:
        try:
            if sigma is not None:
                popt, pcov = curve_fit(model, kappa, c, sigma=sigma, p0=p0)
            else:
                popt, pcov = curve_fit(model, kappa, c, p0=p0)
            
            c_pred = model(kappa, *popt)
            rss = np.sum((c - c_pred) ** 2)
            ss_tot = np.sum((c - np.mean(c)) ** 2)
            R2 = 1 - rss / ss_tot if ss_tot > 0 else 0
            
            # Weighted R² if sigma provided
            if sigma is not None:
                chi2 = np.sum(((c - c_pred) / sigma) ** 2)
                chi2_null = np.sum(((c - np.mean(c)) / sigma) ** 2)
                R2_w = 1 - chi2 / chi2_null if chi2_null > 0 else 0
            else:
                R2_w = R2
            
            results[name] = {
                'a': float(popt[0]), 'b': float(popt[1]),
                'R2_ols': float(R2), 'R2_weighted': float(R2_w),
                'RSS': float(rss)
            }
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # ΔAIC: inv - log (positive = log preferred)
    n = len(kappa)
    if 'RSS' in results.get('log', {}) and 'RSS' in results.get('inv', {}):
        daic_ols = n * np.log(results['inv']['RSS'] / results['log']['RSS'])
        results['DAIC_ols'] = float(daic_ols)
        
        if sigma is not None:
            chi2_log = np.sum(((c - log_model(kappa, results['log']['a'], results['log']['b'])) / sigma) ** 2)
            chi2_inv = np.sum(((c - inv_model(kappa, results['inv']['a'], results['inv']['b'])) / sigma) ** 2)
            daic_w = chi2_inv - chi2_log  # simplified for same k
            results['DAIC_weighted'] = float(daic_w)
    
    if label:
        log_r = results.get('log', {})
        print(f"\n  {label}:")
        print(f"    log: c = {log_r.get('a', '?'):.4f}·ln(κ) + {log_r.get('b', '?'):.4f}, "
              f"R²={log_r.get('R2_weighted', '?'):.4f}")
        print(f"    ΔAIC(inv-log) = {results.get('DAIC_ols', '?')} (OLS), "
              f"{results.get('DAIC_weighted', '?')} (weighted)")
    
    return results

if __name__ == "__main__":
    print("c_kappa_analysis.py — run verify_reproduction.py for full analysis")
