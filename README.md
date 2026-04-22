# Logarithmic Dependence of the OTOC Scaling Exponent on Surface Gravity

**Author:** E. F. Perez-Eugenio  
**ORCID:** [0009-0006-3228-4847](https://orcid.org/0009-0006-3228-4847)

## Summary

This repository contains the data, analysis scripts, and manuscript for a study of how the scaling exponent *c* of out-of-time-order correlators (OTOCs) depends on surface gravity κ in two independent quantum many-body models:

- **SYK model** at 5 finite temperatures (κ = 2πT)
- **Bose-Hubbard sonic analogue** with 7 horizon configurations (κ_sonic)

The central empirical result is:

    c = a · ln(κ) + b

with R² = 0.997 (SYK) and R² = 0.988 (BH sonic), confirmed by ΔAIC analysis favoring logarithmic over inverse dependence.

Additionally, the logistic β-function β(Ω) = −cΩ(1−Ω) produces an emergent AdS₂ geometry with exact Ricci scalar:

    R(Ω) = −2 + 12Ω(1−Ω)

giving R → −2 (AdS₂) at both fixed points of the scrambling flow.

## Repository Structure

```
otoc-surface-gravity/
├── paper/
│   ├── paper5_main_v02.tex       # Manuscript (revtex4-2)
│   └── kaelion_v9.bib            # Bibliography (69 entries, v9.1)
├── data/
│   ├── syk_static_5temps.json    # SYK c(κ), 5 temperatures, N=4-10
│   └── bose_hubbard_sonic_v2.json # BH sonic c(κ), 7 configurations
├── scripts/
│   ├── verify_reproduction.py    # Reproducibility check (run this first)
│   ├── ads2_geometry.py          # AdS₂ curvature analysis
│   ├── c_kappa_analysis.py       # Fits c vs ln(κ): log vs inv vs lin
│   ├── deltaF_analysis.py        # Free energy ΔF vs κ analysis
│   ├── scaling_fits.py           # Ω(N) algebraic fits
│   └── paper5_figures.py         # Generate all figures from data
├── figures/
│   ├── fig1_c_vs_kappa.pdf             # c vs ln(κ) for both models
│   ├── fig2_beta_ads2.pdf              # Emergent AdS₂ geometry R(Ω)
│   ├── fig2_deltaF.pdf                 # Free energy ΔF vs κ (Longo chain)
│   └── fig4_curvature_generalized.pdf  # Uniqueness of logistic β-function
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

## Reproducibility

To verify all key results:

```bash
pip install numpy scipy
cd scripts
python verify_reproduction.py
```

Expected output: all checks PASS, confirming R² values, ΔAIC, and R(Ω→0) = −2.

## Key Results

| Model | a | b | R² | ΔAIC (inv−log) |
|-------|---|---|-----|----------------|
| SYK q=4 (5 temps) | 0.886 | 0.361 | 0.997 | +15.6 (OLS) / +6.9 (weighted) |
| BH sonic (7 configs) | 0.274 | 1.574 | 0.988 | +3.6 (OLS) / +1.0 (weighted) |

## Related Papers

This is Paper V in the Kaelion project series:

- **Paper I:** OTOCs & Scrambling — [DOI: 10.5281/zenodo.18752608](https://doi.org/10.5281/zenodo.18752608)
- **Paper II (1b):** Spatial Scrambling Profiles — [DOI: 10.5281/zenodo.19105623](https://doi.org/10.5281/zenodo.19105623)
- **Paper III:** RG flow — [DOI: 10.5281/zenodo.19211500](https://doi.org/10.5281/zenodo.19211500)

## License

CC BY 4.0 — see [LICENSE](LICENSE).

## Contact

erick.fpe79@gmail.com
