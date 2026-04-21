#!/usr/bin/env python3
"""
================================================================
  p2001_wrpm_check.py — ITU-R P.2001-6 wide-range propagation
                        cross check against BotB canonical P.526
================================================================

WHAT THIS SCRIPT DOES
---------------------
Recomputes basic transmission loss and field strength for every
Knickebein path in the CSV using ITU-R P.2001-6 "Wide Range
Propagation Model" (WRPM), then prints a side-by-side comparison
against:
    (a) the flat Earth free space Friis reference
    (b) the existing P.526-16 smooth Earth diffraction answer

P.2001 bundles four propagation mechanisms into a single answer:
    sub-model 1 = diffraction (Attachment A, = P.526 §3 for smooth sea)
    sub-model 2 = anomalous propagation (ducting / layer reflection)
    sub-model 3 = troposcatter (Attachment E)
    sub-model 4 = sporadic-E reflection (Attachment G)

Sub-models 1 and 3 are implemented directly from the Rec.

Sub-models 2 and 4 are *omitted* because they need global climate
data files (DN_Median.txt, FoEs*.txt) that ship with the Rec.
Including them would require plumbing the ITU data products in.
For the current Knickebein geometry (31.5 MHz VHF, smooth sea,
800 km, midlatitude mid-latitude North Sea) both omitted mechanisms
are expected to contribute near zero to the median-year answer
anyway — sporadic-E is a summer-peak occasional phenomenon, and
ducting above 4000 m receiver altitude puts the ray out of the
boundary layer most of the time.

WHY THIS LIVES IN THE REPO BUT IS NOT IN THE MAIN PIPELINE
----------------------------------------------------------
Run for the Telefunken 800 km case, P.2001 agrees with P.526 to
within the noise of our input values (L_b_combined == L_bm1 because
sub-3 troposcatter comes back many hundreds of dB weaker than
diffraction and Eq 59 picks the stronger mechanism). This confirms
our P.526 baseline but adds no new information to the H0 test,
so the main BotB pipeline continues to use P.526 directly.

If the P.2001 sub-2 (ducting) and sub-4 (sporadic-E) pieces are
ever wired in we'll revisit. For now this script is a read-only
cross check.

================================================================
  USAGE
================================================================
    python3 p2001_wrpm_check.py                # all paths
    python3 p2001_wrpm_check.py TF_800         # one path

================================================================
  REFERENCES
================================================================
  - Recommendation ITU-R P.2001-6 (09/2025), "A general purpose
    wide-range terrestrial propagation model in the frequency range
    30 MHz to 50 GHz"
  - Recommendation ITU-R P.526-16 (10/2025), "Propagation by
    diffraction"
  - botb_itu_analysis.py (../botb_itu_analysis.py) — canonical P.526
    implementation used for sub-model 1 here
"""

import sys
import math
import os

# ================================================================
#   IMPORT CANONICAL P.526 IMPLEMENTATION FROM PARENT DIRECTORY
# ================================================================
# The parent directory holds botb_itu_analysis.py with the
# validated P.526-16 first-term residue series. P.2001 Attachment A
# uses the exact same math (same K, β, X, Y, F_X, G(Y) formulas and
# the same G(Y) ≥ 2 + 20·log₁₀(K) clamp), so we reuse that function
# for sub-model 1 rather than reimplement from the Rec.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
from botb_itu_analysis import itu_diffraction_loss   # noqa: E402

# ================================================================
#   CANONICAL BOTB CONSTANTS (match botb_itu_analysis.py)
# ================================================================
PT_W          = 3000.0              # Large Knickebein TX power (Telefunken)
PT_DBW        = 10.0 * math.log10(PT_W)
GT_DBI        = 26.0                # Knickebein 99 x 29 m aperture directivity
NOISE_FLOOR   = -151.2              # dBW, ITU-R P.372 galactic at 31.5 MHz
CROSSOVER_DB  = 19.87               # 5 deg squint equisignal crossover loss (sinc² derivation)
A_E_KM        = 8495.0              # 4/3 Earth, k=4/3, ITU-R P.453

# ================================================================
#   P.2001 ATTACHMENT E — TROPOSCATTER CLIMATE PARAMETERS
# ================================================================
# From Table E.1 of ITU-R P.2001-6. Climate zone code 0 = "Sea path"
# which applies to all Knickebein over-sea paths. Land paths would
# select zone 2 (temperate continental) at these latitudes but would
# need the TropoClim.txt lookup to be certain. For land paths we
# default to zone 2 here as a reasonable mid-latitude proxy.
#
# gamma is an atmospheric structure scale (km^-1), M is the
# climate's base transmission loss offset (dB).
TROPO_SEA  = dict(M=116.00, gamma=20.27, Y90_eq="E.7")
TROPO_LAND = dict(M=119.73, gamma=20.27, Y90_eq="E.6")   # zone 2 proxy

# ================================================================
#   CSV LOCATION
# ================================================================
CSV_PATH = os.path.join(_here, "knickebein_paths.csv")


# ================================================================
#   HELPERS
# ================================================================
def lbfs_dB(f_MHz, d_km):
    """Free space basic transmission loss (ITU standard form)."""
    return 32.44 + 20.0 * math.log10(f_MHz) + 20.0 * math.log10(d_km)


def E_dBuVm(Pt_dBW, Gt_dBi, f_MHz, Lb_dB):
    """
    Field strength at an isotropic reference point, in dBuV/m.
    Standard CCIR relationship that follows from S = EIRP/(4π d²)
    and E² = 120π·S. Independent of RX gain because E describes the
    incident field, not the power captured by a particular antenna.
    """
    return Pt_dBW + Gt_dBi + 20.0 * math.log10(f_MHz) - Lb_dB + 107.22


def dBuVm_to_uVm(x):
    return 10.0 ** (x / 20.0)


def p2001_submodel3_sea(d_km, h_tx_m, h_rx_m, f_MHz, Gt_dBi, Gr_dBi,
                        a_e_km, climate):
    """
    P.2001 Attachment E troposcatter basic transmission loss.
    Smooth earth approximation for scatter angle. p = 50% (median).

    Parameters
    ----------
    d_km      : great circle distance
    h_tx_m    : TX height ASL
    h_rx_m    : RX height ASL
    f_MHz     : frequency
    Gt_dBi,Gr_dBi : antenna gains (enter L_coup term)
    a_e_km    : effective Earth radius
    climate   : TROPO_SEA or TROPO_LAND dict

    Returns
    -------
    Lbs       : troposcatter basic transmission loss (dB)
    """
    M     = climate["M"]
    gamma = climate["gamma"]

    # --- scatter geometry (smooth Earth approximation) ---
    # theta_e = d / a_e is the angle subtended at the Earth's centre
    # by the great circle path. Each terminal's horizon dip sqrt(2h/a_e)
    # reduces the scatter angle because the ray does not climb above
    # the local horizontal before meeting the common volume.
    theta_e_rad = d_km / a_e_km
    t_dip_rad   = math.sqrt(2.0 * h_tx_m / (a_e_km * 1000.0))
    r_dip_rad   = math.sqrt(2.0 * h_rx_m / (a_e_km * 1000.0))
    theta_mrad  = max(
        (theta_e_rad - t_dip_rad - r_dip_rad) * 1000.0,
        1e-6
    )

    # --- common volume geometry (Eqs E.3, E.4) ---
    H      = 0.25e-3 * theta_mrad * d_km
    h_trop = 0.125e-6 * (theta_mrad ** 2) * a_e_km

    # --- Eq E.2 L_N (common volume height dependent loss) ---
    L_N = 20.0 * math.log10(5.0 + gamma * H) + 4.34 * gamma * h_trop

    # --- Eq E.13 L_dist (distance dependent loss) ---
    L_dist = max(
        10.0 * math.log10(d_km) + 30.0 * math.log10(theta_mrad) + L_N,
        20.0 * math.log10(d_km) + 0.573 * theta_mrad + 20.0
    )

    # --- Eq E.14 L_freq (f in GHz per Rec conventions in Att E) ---
    f_GHz  = f_MHz / 1000.0
    L_freq = 25.0 * math.log10(f_GHz) - 2.5 * (math.log10(0.5 * f_GHz)) ** 2

    # --- Eq E.15 aperture to medium coupling loss ---
    L_coup = 0.07 * math.exp(0.055 * (Gt_dBi + Gr_dBi))

    # --- Eq E.12 Y_p at p=50% is zero (median) ---
    Y_p = 0.0

    # --- Eq E.16 ---
    Lbs = M + L_freq + L_dist + L_coup - Y_p

    # --- Eq E.17 floor: troposcatter can never beat free space ---
    Lbs = max(Lbs, lbfs_dB(f_MHz, d_km))
    return Lbs


def combine_eq59(Lbm1, Lbm3):
    """
    P.2001 Equation 59: combine sub models 1 and 3 (uncorrelated
    approximation at a single time percentage).

    This is NOT the full Eq 59. The full version combines the
    already-correlated (1+2) with 3 and 4. With only sub 1 and
    sub 3 available we use the two-term reduction, which is
    conservative: omitting sub 2 and sub 4 can only bias Lb toward
    higher loss.
    """
    Lm = min(Lbm1, Lbm3)
    inner = 10.0 ** (-0.2 * (Lbm1 - Lm)) + 10.0 ** (-0.2 * (Lbm3 - Lm))
    return Lm - 5.0 * math.log10(inner)


# ================================================================
#   PATH LOADER
# ================================================================
def load_paths(only_id=None):
    paths = []
    with open(CSV_PATH) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.strip().split(",")
            if cols[0] == "path_id":
                continue
            if only_id and cols[0] != only_id:
                continue
            paths.append(cols)
    return paths


# ================================================================
#   MAIN COMPUTATION
# ================================================================
def analyse_path(cols):
    pid    = cols[0]
    d_m    = float(cols[12])
    h_tx   = float(cols[6])
    h_rx   = float(cols[9])
    f_MHz  = float(cols[10])
    gr     = float(cols[11])
    ground = cols[14]
    d_km   = d_m / 1000.0

    # --- Flat Earth reference ---
    Lbfs      = lbfs_dB(f_MHz, d_km)
    E_fe_dB   = E_dBuVm(PT_DBW, GT_DBI, f_MHz, Lbfs)

    # --- Globe sub-model 1 (diffraction = P.526) ---
    diff = itu_diffraction_loss(
        d_m, h_tx, h_rx, freq=f_MHz * 1e6, a_e=A_E_KM * 1000.0,
        ground=ground, polarization="vertical",
    )
    Lbm1     = Lbfs + diff["loss_dB"]
    E_diff_dB = E_dBuVm(PT_DBW, GT_DBI, f_MHz, Lbm1)

    # --- Globe sub-model 3 (troposcatter) ---
    climate = TROPO_SEA if ground == "sea" else TROPO_LAND
    Lbm3    = p2001_submodel3_sea(
        d_km, h_tx, h_rx, f_MHz, GT_DBI, gr, A_E_KM, climate
    )
    E_tropo_dB = E_dBuVm(PT_DBW, GT_DBI, f_MHz, Lbm3)

    # --- Combined (Eq 59, sub 1 + sub 3) ---
    Lb         = combine_eq59(Lbm1, Lbm3)
    E_comb_dB  = E_dBuVm(PT_DBW, GT_DBI, f_MHz, Lb)

    # --- SNR and detect margin (uses RX gain, unlike E) ---
    Pr_fe    = PT_DBW + GT_DBI + gr - Lbfs
    Pr_globe = PT_DBW + GT_DBI + gr - Lb
    SNR_fe    = Pr_fe    - NOISE_FLOOR
    SNR_globe = Pr_globe - NOISE_FLOOR

    return dict(
        pid=pid, d_km=d_km, h_tx=h_tx, h_rx=h_rx, Gr=gr,
        ground=ground,
        Lbfs=Lbfs, Lbm1=Lbm1, Lbm3=Lbm3, Lb=Lb,
        E_fe=E_fe_dB, E_diff=E_diff_dB, E_tropo=E_tropo_dB, E_comb=E_comb_dB,
        SNR_fe=SNR_fe, SNR_globe=SNR_globe,
    )


# ================================================================
#   OUTPUT FORMATTING
# ================================================================
def print_single(r):
    print("=" * 78)
    print(f"  P.2001-6 WRPM cross check — {r['pid']}")
    print("=" * 78)
    print(f"  d = {r['d_km']:.0f} km | h_tx = {r['h_tx']:.0f} m | "
          f"h_rx = {r['h_rx']:.0f} m | Gr = {r['Gr']:.0f} dBi | ground = {r['ground']}")
    print()
    print(f"  {'Model':<40s} {'Lb (dB)':>8s} {'E (dBuV/m)':>12s} {'E (uV/m)':>14s}")
    print("  " + "-" * 76)
    print(f"  {'Flat Earth (free space)':<40s} "
          f"{r['Lbfs']:>8.1f} {r['E_fe']:>+12.2f} {dBuVm_to_uVm(r['E_fe']):>14.4g}")
    print(f"  {'Globe sub-1 (diffraction, Att A)':<40s} "
          f"{r['Lbm1']:>8.1f} {r['E_diff']:>+12.2f} {dBuVm_to_uVm(r['E_diff']):>14.4g}")
    print(f"  {'Globe sub-3 (troposcatter, Att E)':<40s} "
          f"{r['Lbm3']:>8.1f} {r['E_tropo']:>+12.2f} {dBuVm_to_uVm(r['E_tropo']):>14.4g}")
    print(f"  {'Globe COMBINED (Eq 59, sub 1+3)':<40s} "
          f"{r['Lb']:>8.1f} {r['E_comb']:>+12.2f} {dBuVm_to_uVm(r['E_comb']):>14.4g}")
    print()
    print(f"  FE  peak SNR = {r['SNR_fe']:+.1f} dB")
    print(f"  Globe peak SNR = {r['SNR_globe']:+.1f} dB")
    print(f"  FE / Globe field strength ratio = {dBuVm_to_uVm(r['E_fe'])/dBuVm_to_uVm(r['E_comb']):.2e}  "
          f"({r['E_fe']-r['E_comb']:.1f} dB)")
    print()
    print("  Omitted: sub-2 (ducting), sub-4 (sporadic-E).")
    print("  Both need ITU climate data files; expected contribution small")
    print("  for median-year midlatitude VHF paths.")


def print_summary(results):
    print("=" * 92)
    print(f"  {'path':<14} {'d':>5} {'ground':>6}  "
          f"{'FE uV/m':>10} {'Globe uV/m':>12} {'ratio dB':>10} "
          f"{'FE SNR':>8} {'GL SNR':>8}")
    print("=" * 92)
    for r in results:
        ratio_dB = r['E_fe'] - r['E_comb']
        print(f"  {r['pid']:<14} {r['d_km']:>5.0f} {r['ground']:>6}  "
              f"{dBuVm_to_uVm(r['E_fe']):>10.3g} "
              f"{dBuVm_to_uVm(r['E_comb']):>12.3g} "
              f"{ratio_dB:>+10.1f} "
              f"{r['SNR_fe']:>+8.1f} {r['SNR_globe']:>+8.1f}")


# ================================================================
#   ENTRY POINT
# ================================================================
def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    paths = load_paths(only_id=only)
    if not paths:
        print(f"No paths matched filter {only!r}.")
        sys.exit(1)

    results = [analyse_path(cols) for cols in paths]

    if len(results) == 1:
        print_single(results[0])
    else:
        print_summary(results)


if __name__ == "__main__":
    main()
