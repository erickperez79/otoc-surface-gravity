"""
deltaF_analysis.py — Free energy ΔF vs κ analysis
==================================================
Tests ΔF ∝ κ·ln(κ) (Longo chain prediction) against
ΔF ∝ κ and ΔF ∝ κ² alternatives.

Usage: python deltaF_analysis.py
"""
import numpy as np
from scipy.optimize import curve_fit
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def kappa_logkappa(kappa, alpha):
    """ΔF = α · κ · ln(κ)"""
    return alpha * kappa * np.log(kappa)

def kappa_linear(kappa, alpha):
    """ΔF = α · κ"""
    return alpha * kappa

def kappa_squared(kappa, alpha):
    """ΔF = α · κ²"""
    return alpha * kappa ** 2

def analyze_deltaF(kappas, deltaF_vals, label=""):
    """Compare three models for ΔF(κ)."""
    kappa = np.array(kappas)
    dF = np.array(deltaF_vals)
    
    results = {}
    for name, model, p0 in [('kappa_logkappa', kappa_logkappa, [1.0]),
                              ('kappa_linear', kappa_linear, [1.0]),
                              ('kappa_squared', kappa_squared, [0.1])]:
        try:
            popt, _ = curve_fit(model, kappa, dF, p0=p0)
            dF_pred = model(kappa, *popt)
            rss = np.sum((dF - dF_pred) ** 2)
            ss_tot = np.sum((dF - np.mean(dF)) ** 2)
            R2 = 1 - rss / ss_tot if ss_tot > 0 else 0
            n = len(kappa)
            aic = n * np.log(rss / n) + 2 * 1  # k=1 parameter
            results[name] = {'alpha': float(popt[0]), 'R2': float(R2), 'AIC': float(aic)}
        except Exception as e:
            results[name] = {'error': str(e)}
    
    if label:
        print(f"\n  {label}:")
        for name in ['kappa_logkappa', 'kappa_linear', 'kappa_squared']:
            r = results.get(name, {})
            print(f"    {name}: α={r.get('alpha', '?'):.4f}, R²={r.get('R2', '?'):.4f}")
    
    return results

if __name__ == "__main__":
    print("deltaF_analysis.py — run verify_reproduction.py for full analysis")
