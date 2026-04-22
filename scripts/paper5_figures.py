"""
paper5_figures.py — Generate all Paper V figures from data
==========================================================
Reads from authoritative JSONs and produces publication-quality PDFs.

Usage: python paper5_figures.py
Output: ../figures/fig1_c_vs_kappa.pdf, ../figures/fig2_beta_ads2.pdf
"""
import numpy as np
from scipy.optimize import curve_fit
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — cannot generate figures")
    sys.exit(1)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def log_model(kappa, a, b):
    return a * np.log(kappa) + b

# ============================================================
# FIGURE 1: c vs ln(κ) for SYK and BH sonic
# ============================================================
def make_fig1():
    print("Generating fig1_c_vs_kappa.pdf...")
    
    syk_file = os.path.join(DATA_DIR, 'syk_static_5temps.json')
    bh_file = os.path.join(DATA_DIR, 'bose_hubbard_sonic_v2.json')
    
    if not os.path.exists(syk_file) or not os.path.exists(bh_file):
        print(f"  Data files not found in {DATA_DIR}/")
        return
    
    with open(syk_file) as f:
        syk = json.load(f)
    with open(bh_file) as f:
        bh = json.load(f)
    
    # Extract data (handle both possible JSON structures)
    def extract_kappa_c(data, model_name):
        kappas, cs, errs = [], [], []
        entries = data if isinstance(data, list) else data.get('temperatures', data.get('configurations', []))
        for e in entries:
            k = e.get('kappa', e.get('kappa_sonic'))
            c = e.get('c')
            ce = e.get('c_err', e.get('sigma_c', 0.1))
            if k is not None and c is not None:
                kappas.append(k)
                cs.append(c)
                errs.append(ce)
        return np.array(kappas), np.array(cs), np.array(errs)
    
    k_syk, c_syk, e_syk = extract_kappa_c(syk, "SYK")
    k_bh, c_bh, e_bh = extract_kappa_c(bh, "BH")
    
    # Fits
    p_syk, _ = curve_fit(log_model, k_syk, c_syk, sigma=e_syk, p0=[0.9, 0.4])
    p_bh, _ = curve_fit(log_model, k_bh, c_bh, sigma=e_bh, p0=[0.3, 1.5])
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    kappa_range_syk = np.linspace(min(k_syk) * 0.8, max(k_syk) * 1.2, 100)
    kappa_range_bh = np.linspace(min(k_bh) * 0.8, max(k_bh) * 1.2, 100)
    
    ax.errorbar(k_syk, c_syk, yerr=e_syk, fmt='o', color='#d62728', 
                markersize=8, capsize=4, label=f'SYK $q=4$: $c = {p_syk[0]:.3f}\\ln\\kappa + {p_syk[1]:.3f}$')
    ax.plot(kappa_range_syk, log_model(kappa_range_syk, *p_syk), '-', color='#d62728', alpha=0.5)
    
    ax.errorbar(k_bh, c_bh, yerr=e_bh, fmt='s', color='#1f77b4',
                markersize=7, capsize=4, label=f'BH sonic: $c = {p_bh[0]:.3f}\\ln\\kappa + {p_bh[1]:.3f}$')
    ax.plot(kappa_range_bh, log_model(kappa_range_bh, *p_bh), '-', color='#1f77b4', alpha=0.5)
    
    ax.set_xscale('log')
    ax.set_xlabel(r'$\kappa$ (surface gravity)', fontsize=12)
    ax.set_ylabel(r'$c$ (scaling exponent)', fontsize=12)
    ax.set_title(r'$c \propto \ln\kappa$ in two independent models', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, 'fig1_c_vs_kappa.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()

# ============================================================
# FIGURE 2: AdS₂ geometry
# ============================================================
def make_fig2():
    print("Generating fig2_beta_ads2.pdf...")
    
    omega = np.linspace(0.001, 0.999, 1000)
    R = -2 + 12 * omega * (1 - omega)
    
    omega_cross_1 = (3 - np.sqrt(3)) / 6
    omega_cross_2 = (3 + np.sqrt(3)) / 6
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(omega, R, 'b-', linewidth=2, label=r'$R(\Omega) = -2 + 12\Omega(1-\Omega)$')
    ax.axhline(y=-2, color='r', linestyle='--', alpha=0.5, label=r'$R = -2$ (AdS$_2$)')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    
    ax.fill_between(omega, R, -2, where=(R > -2), alpha=0.08, color='orange')
    ax.fill_between(omega, -2, np.minimum(R, -2), where=(R < -2), alpha=0.05, color='blue')
    
    ax.annotate(r'AdS$_2$ (IR)', xy=(0.02, -1.95), fontsize=10, color='red')
    ax.annotate(r'AdS$_2$ (UV)', xy=(0.88, -1.95), fontsize=10, color='red')
    ax.annotate(r'dS-like', xy=(0.45, 0.7), fontsize=10, color='orange')
    
    ax.set_xlabel(r'$\Omega$ (scrambling parameter)', fontsize=12)
    ax.set_ylabel(r'$R$ (Ricci scalar)', fontsize=12)
    ax.set_title(r'Emergent geometry from $\beta(\Omega) = -c\Omega(1-\Omega)$', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-2.5, 1.5)
    
    fig.tight_layout()
    outpath = os.path.join(FIG_DIR, 'fig2_beta_ads2.pdf')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()

if __name__ == "__main__":
    make_fig1()
    make_fig2()
    print("\nAll figures generated.")
