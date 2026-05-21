# Equation references for notebook 004 (Processing-Gain Ceiling)

Primary sources backing every equation used in `jupyter_book/004_Processing_Gain_Ceiling.ipynb`.

| Equation in 004 | Source file | Where to verify (exact page) |
|---|---|---|
| Thermal noise `kTB` (the thermal term of the floor) | `Nyquist_1928_Thermal_Agitation_of_Electric_Charge.pdf` | Nyquist, H. (1928). Physical Review 32, 110-113. Mean-square thermal EMF, equation (4), p. 113. |
| Noise factor concept (origin) | `RCA_Review_1942-01_North_Absolute_Sensitivity_pp332-343.pdf` | North, D.O. (1942). The Absolute Sensitivity of Radio Receivers. RCA Review 6(3), 332-343. Noise factor `N`, equation (4), p. 335. |
| Noise factor `F` (receiver-noise term, rigorous definition) | `ProcIRE_1944-07_Friis_Noise_Figures_pp419-422.pdf` | Friis, H.T. (1944). Noise Figures of Radio Receivers. Proc. IRE 32(7), 419-422. Noise figure definition, equation (4), p. 420; also `kT df` and the 290 K standard, p. 420. |
| Optimal-detector basis (likelihood-ratio / most powerful test) | `Neyman_Pearson_1933_Most_Efficient_Tests.pdf` | Neyman, J. & Pearson, E.S. (1933). Phil. Trans. Roy. Soc. A 231, 289-337. Best-critical-region criterion, p. 300. |
| Galactic radio noise `Fa = 52 - 23 log10(f_MHz)` (baseline floor) | `ITU-R_P.372-16_Radio_Noise.pdf` | ITU-R P.372-16, Part 4 §4.1, equation (13), p. 16. |
| Fock smooth-Earth diffraction loss (baseline GE SNR) | `ITU-R_P.526-16_Propagation_by_Diffraction.pdf` | ITU-R P.526-16 (11/2025), §3.1.1.2, equations 13-19, pp. 9-10. |
| Matched filter optimality, output SNR = 2E/N0 | `ProcIEEE_1963_North_Matched_Filter_pp1016-1027.pdf` | North, D.O. (1943). RCA Tech. Report PTR-6C; reprinted Proc. IEEE 51(7), 1016-1027 (1963). Matched-filter result at reprint p. 1021, §A, equations 24-28. |

## Equations that are algebra, not separate citations

- `G_proc = 10 log10(B_ref / B_det)` is the definition of processing gain.
- `B_det = 1/(2 tau)` is the noise-equivalent bandwidth of an integrate-and-dump
  matched filter (derivable: ENBW = integral of |H(f)|^2 / |H(0)|^2 = 1/(2 tau)
  for a rectangular integration window).
- `G_proc = 10 log10(2 tau B_ref)` is the two above combined.
- `tau_req = 10^(D/10) / (2 B_ref)` is that result inverted for tau.
