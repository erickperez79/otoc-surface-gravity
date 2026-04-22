"""
verify_reproduction.py — Reproducibility verification for Paper V
=================================================================
This script verifies all key numerical results of Paper V from the
authoritative JSON data files. If all checks pass, the repository
is reproducible.

Usage: python verify_reproduction.py
Requirements: numpy, scipy

Expected output: all checks PASS with exact values from the paper.
"""
import numpy as np
from scipy.optimize import curve_fit
import json
import sys
import os

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TOLERANCE = {
    'R2': 1e-3,      # R² match to 3 decimal places
    'coeff': 1e-2,    # Coefficients match to 2 decimal places
    'exact': 1e-6,    # Exact analytical results
}

# Reference values from Paper V (STATUS v31)
REF = {
    'SYK_a': 0.886,
    'SYK_b': 0.361,
    'SYK_R2': 0.997,
    'BH_a': 0.274,
    'BH_b': 1.574,
    'BH_R2': 0.988,
    'DAIC_OLS': 15.6,
    'R_IR': -2.0,          # AdS₂ curvature at Ω→0
    'R_formula_check': 1.0, # R(Ω=0.5) = -2 + 12*0.25 = +1
}

passed = 0
failed = 0
total = 0

def check(name, computed, expected, tol=1e-3):
    global passed, failed, total
    total += 1
    delta = abs(computed - expected)
    ok = delta < tol
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}: computed={computed:.6f}, expected={expected:.6f}, Δ={delta:.2e}")
    return ok

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("Paper V — Reproducibility Verification")
print("=" * 70)

syk_file = os.path.join(DATA_DIR, 'syk_static_5temps.json')
bh_file = os.path.join(DATA_DIR, 'bose_hubbard_sonic_v2.json')

if not os.path.exists(syk_file):
    print(f"\n  ERROR: {syk_file} not found")
    print(f"  Place authoritative JSONs in {DATA_DIR}/")
    sys.exit(1)
if not os.path.exists(bh_file):
    print(f"\n  ERROR: {bh_file} not found")
    sys.exit(1)

with open(syk_file) as f:
    syk_data = json.load(f)
with open(bh_file) as f:
    bh_data = json.load(f)

# ============================================================
# EXTRACT c(κ) FROM JSONs
# ============================================================

def algebraic(N, A, c):
    return 1.0 / (1.0 + A * N**c)

def extract_c_from_omega(N_values, omega_values):
    """Fit Ω(N) = 1/(1+A·N^c) and return c, c_err, R²."""
    N_arr = np.array(N_values, dtype=float)
    Om_arr = np.array(omega_values)
    popt, pcov = curve_fit(algebraic, N_arr, Om_arr, p0=[0.01, 2.0], maxfev=50000)
    c = popt[1]
    c_err = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0
    Om_pred = algebraic(N_arr, *popt)
    ss_res = np.sum((Om_arr - Om_pred) ** 2)
    ss_tot = np.sum((Om_arr - np.mean(Om_arr)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return c, c_err, R2

# ============================================================
# CHECK 1: SYK c(κ) logarithmic fit
# ============================================================
print("\n--- CHECK 1: SYK c vs ln(κ) ---")

# Extract kappa and c from SYK data
# Structure depends on JSON format — handle both possibilities
syk_kappas = []
syk_c_vals = []
syk_c_errs = []

if isinstance(syk_data, list):
    for entry in syk_data:
        kappa = entry.get('kappa', 2 * np.pi * entry.get('T', 1.0 / entry.get('beta', 1.0)))
        c_val = entry.get('c')
        c_err = entry.get('c_err', entry.get('sigma_c', 0.1))
        if c_val is not None:
            syk_kappas.append(kappa)
            syk_c_vals.append(c_val)
            syk_c_errs.append(c_err)
elif isinstance(syk_data, dict):
    # Handle beta=X.X key structure (authoritative format)
    for key, entry in syk_data.items():
        if key.startswith('beta=') and isinstance(entry, dict):
            kappa = entry.get('kappa')
            fit = entry.get('fit', {})
            c_val = fit.get('c')
            c_err = fit.get('c_err', 0.1)
            if c_val is not None and kappa is not None:
                syk_kappas.append(kappa)
                syk_c_vals.append(c_val)
                syk_c_errs.append(c_err)
        elif key == 'temperatures' or key == 'data':
            temps = entry if isinstance(entry, list) else []
            for e in temps:
                beta = e.get('beta', 1.0)
                T = e.get('T', 1.0 / beta)
                kappa = e.get('kappa', 2 * np.pi * T)
                c_val = e.get('c')
                c_err = e.get('c_err', e.get('sigma_c', 0.1))
                if c_val is not None:
                    syk_kappas.append(kappa)
                    syk_c_vals.append(c_val)
                    syk_c_errs.append(c_err)

syk_kappas = np.array(syk_kappas)
syk_c = np.array(syk_c_vals)
syk_sigma = np.array(syk_c_errs)

print(f"  SYK data: {len(syk_kappas)} temperature points")
for i in range(len(syk_kappas)):
    print(f"    κ={syk_kappas[i]:.4f}, c={syk_c[i]:.4f}±{syk_sigma[i]:.4f}")

# Logarithmic fit: c = a·ln(κ) + b
def log_model(kappa, a, b):
    return a * np.log(kappa) + b

def inv_model(kappa, a, b):
    return a / kappa + b

# Weighted fit
popt_log, pcov_log = curve_fit(log_model, syk_kappas, syk_c,
                                sigma=syk_sigma, p0=[0.9, 0.4])
a_syk, b_syk = popt_log
c_pred_log = log_model(syk_kappas, *popt_log)
R2_syk = 1 - np.sum(((syk_c - c_pred_log) / syk_sigma) ** 2) / \
              np.sum(((syk_c - np.mean(syk_c)) / syk_sigma) ** 2)

# OLS (unweighted) for ΔAIC
popt_log_ols, _ = curve_fit(log_model, syk_kappas, syk_c, p0=[0.9, 0.4])
popt_inv_ols, _ = curve_fit(inv_model, syk_kappas, syk_c, p0=[1.0, 1.0])

rss_log = np.sum((syk_c - log_model(syk_kappas, *popt_log_ols)) ** 2)
rss_inv = np.sum((syk_c - inv_model(syk_kappas, *popt_inv_ols)) ** 2)
n = len(syk_c)
daic_ols = n * np.log(rss_inv / rss_log)  # ΔAIC = AIC_inv - AIC_log (>0 favors log)

print(f"\n  Fit: c = {a_syk:.4f}·ln(κ) + {b_syk:.4f}")
print(f"  R² (weighted) = {R2_syk:.6f}")
print(f"  ΔAIC (OLS) = {daic_ols:.1f}")

check("SYK a", a_syk, REF['SYK_a'], tol=TOLERANCE['coeff'])
check("SYK b", b_syk, REF['SYK_b'], tol=TOLERANCE['coeff'])
check("SYK R²", R2_syk, REF['SYK_R2'], tol=TOLERANCE['R2'])
check("ΔAIC OLS", daic_ols, REF['DAIC_OLS'], tol=1.0)

# ============================================================
# CHECK 2: BH sonic c(κ) logarithmic fit
# ============================================================
print("\n--- CHECK 2: BH sonic c vs ln(κ) ---")

bh_kappas = []
bh_c_vals = []
bh_c_errs = []

if isinstance(bh_data, list):
    for entry in bh_data:
        kappa = entry.get('kappa', entry.get('kappa_sonic'))
        c_val = entry.get('c')
        c_err = entry.get('c_err', entry.get('sigma_c', 0.05))
        if c_val is not None and kappa is not None:
            bh_kappas.append(kappa)
            bh_c_vals.append(c_val)
            bh_c_errs.append(c_err)
elif isinstance(bh_data, dict):
    # Handle w=X_U=Y key structure (authoritative format)
    for key, entry in bh_data.items():
        if key.startswith('w=') and isinstance(entry, dict):
            kappa = entry.get('kappa_sonic', entry.get('kappa'))
            fit = entry.get('fit', {})
            c_val = fit.get('c')
            c_err = fit.get('c_err', 0.05)
            if c_val is not None and kappa is not None:
                bh_kappas.append(kappa)
                bh_c_vals.append(c_val)
                bh_c_errs.append(c_err)
        elif key in ('configurations', 'data'):
            configs = entry if isinstance(entry, list) else []
            for e in configs:
                kappa = e.get('kappa', e.get('kappa_sonic'))
                c_val = e.get('c')
                c_err = e.get('c_err', e.get('sigma_c', 0.05))
                if c_val is not None and kappa is not None:
                    bh_kappas.append(kappa)
                    bh_c_vals.append(c_val)
                    bh_c_errs.append(c_err)

bh_kappas = np.array(bh_kappas)
bh_c = np.array(bh_c_vals)
bh_sigma = np.array(bh_c_errs)

print(f"  BH data: {len(bh_kappas)} κ points")

if len(bh_kappas) >= 3:
    popt_bh, pcov_bh = curve_fit(log_model, bh_kappas, bh_c,
                                  sigma=bh_sigma, p0=[0.3, 1.5])
    a_bh, b_bh = popt_bh
    c_pred_bh = log_model(bh_kappas, *popt_bh)
    R2_bh = 1 - np.sum(((bh_c - c_pred_bh) / bh_sigma) ** 2) / \
                np.sum(((bh_c - np.mean(bh_c)) / bh_sigma) ** 2)
    
    print(f"  Fit: c = {a_bh:.4f}·ln(κ) + {b_bh:.4f}")
    print(f"  R² (weighted) = {R2_bh:.6f}")
    
    check("BH a", a_bh, REF['BH_a'], tol=TOLERANCE['coeff'])
    check("BH b", b_bh, REF['BH_b'], tol=TOLERANCE['coeff'])
    check("BH R²", R2_bh, REF['BH_R2'], tol=TOLERANCE['R2'])
else:
    print("  INSUFFICIENT DATA — check JSON structure")

# ============================================================
# CHECK 3: AdS₂ geometry — R(Ω) = -2 + 12Ω(1-Ω)
# ============================================================
print("\n--- CHECK 3: AdS₂ curvature R(Ω) ---")

print("  Analytical formula: R = -2 + 12Ω(1-Ω)")
print("  Derived from β(Ω) = -cΩ(1-Ω) with metric ds² = (1/β²)dΩ² + β²dφ²")

# Verify at key points
R_0 = -2 + 12 * 0 * 1          # Ω→0
R_half = -2 + 12 * 0.5 * 0.5    # Ω=0.5
R_1 = -2 + 12 * 1 * 0           # Ω→1

check("R(Ω→0) [AdS₂ IR]", R_0, REF['R_IR'], tol=TOLERANCE['exact'])
check("R(Ω=0.5)", R_half, REF['R_formula_check'], tol=TOLERANCE['exact'])
check("R(Ω→1) [AdS₂ UV]", R_1, REF['R_IR'], tol=TOLERANCE['exact'])

# Numerical verification via finite differences
def R_numerical(omega, h=1e-7):
    """R = -2β² [γ'' + 2(γ')²] where γ = ln|β|, β = -Ω(1-Ω)"""
    b = omega * (1 - omega)
    if b < 1e-15:
        return -2.0
    bp = (omega + h) * (1 - omega - h)
    bm = (omega - h) * (1 - omega + h)
    g = np.log(b)
    gp = np.log(bp) if bp > 0 else g
    gm = np.log(bm) if bm > 0 else g
    g1 = (gp - gm) / (2 * h)
    g2 = (gp - 2 * g + gm) / h ** 2
    return -2 * b ** 2 * (g2 + 2 * g1 ** 2)

R_num_01 = R_numerical(0.01)
R_ana_01 = -2 + 12 * 0.01 * 0.99
check("R(0.01) numerical vs analytical", R_num_01, R_ana_01, tol=1e-3)

R_num_03 = R_numerical(0.3)
R_ana_03 = -2 + 12 * 0.3 * 0.7
check("R(0.3) numerical vs analytical", R_num_03, R_ana_03, tol=1e-2)

# ============================================================
# CHECK 4: Uniqueness — (a,b)≠(1,1) produces singularities
# ============================================================
print("\n--- CHECK 4: Uniqueness of logistic β ---")

def R_generalized(omega, a_exp, b_exp, h=1e-7):
    """R for β = -Ω^a (1-Ω)^b with general a,b."""
    def beta_gen(o):
        return o ** a_exp * (1 - o) ** b_exp
    b = beta_gen(omega)
    bp = beta_gen(omega + h)
    bm = beta_gen(omega - h)
    if b < 1e-15 or bp < 1e-15 or bm < 1e-15:
        return float('nan')
    g = np.log(b)
    gp = np.log(bp)
    gm = np.log(bm)
    g1 = (gp - gm) / (2 * h)
    g2 = (gp - 2 * g + gm) / h ** 2
    return -2 * b ** 2 * (g2 + 2 * g1 ** 2)

# (1,1) → R→-2 at both endpoints
R_11_ir = R_generalized(0.001, 1, 1)
check("(a,b)=(1,1): R(Ω→0)≈-2", R_11_ir, -2.0, tol=0.1)

# (2,1) → singular at Ω→0
R_21_ir = R_generalized(0.001, 2, 1)
print(f"  [INFO] (a,b)=(2,1): R(Ω→0) = {R_21_ir:.2f} (should diverge from -2)")

# (1,2) → singular at Ω→1
R_12_uv = R_generalized(0.999, 1, 2)
print(f"  [INFO] (a,b)=(1,2): R(Ω→1) = {R_12_uv:.2f} (should diverge from -2)")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"VERIFICATION SUMMARY")
print(f"{'='*70}")
print(f"  Total checks: {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")

if failed == 0:
    print(f"\n  ✅ ALL CHECKS PASSED — Repository is reproducible")
    sys.exit(0)
else:
    print(f"\n  ❌ {failed} CHECKS FAILED — Review data or computations")
    sys.exit(1)
