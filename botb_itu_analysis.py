#!/usr/bin/env python3
"""
Battle of the Beams — ITU-R P.526 Propagation Analysis
=======================================================

Recomputed from scratch using the ITU-R P.526-16 (2025) smooth Earth
diffraction standard.  This replaces the previous G(Y) = sqrt(1+piY)
engineering approximation with the internationally standardised height
gain formula.

The previous code used a non-standard height gain formula that
underestimated the signal at aircraft altitude by ~45 dB.  This script
uses:
  1. ITU-R P.526-16 Eq. 13-18 for smooth Earth diffraction
  2. Shatz/MIT 35-term Fock residue series for cross-validation
  3. Friis FSPL for the flat Earth baseline

All paths are loaded from knickebein_paths.csv so the data is
separated from the logic and can be adjusted without touching code.

References:
  [1] ITU-R P.526-16 (2025). "Propagation by diffraction."
      International Telecommunication Union. Section 3.1.
  [2] Fock, V.A. (1965). Electromagnetic Diffraction and Propagation
      Problems. Pergamon Press. Ch. 10, Eq. (6.10).
  [3] Shatz, M.P. and G.H. Polychronopoulos (1988). "An Improved
      Spherical Earth Diffraction Algorithm for SEKE." MIT Lincoln
      Lab, Project Report CMT-111. ADA195847.
  [4] Friis, H.T. (1946). "A Note on a Simple Transmission Formula."
      Proc. IRE 34(5), 254-256.
  [5] Bauer, A.O. (2004). "Some historical and technical aspects of
      radio navigation in Germany, 1907-1945."
"""

import numpy as np
import csv
import os
from scipy.special import airy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
#  PHYSICAL CONSTANTS
# ================================================================

# Speed of light in vacuum (m/s). Exact by SI definition.
C = 299_792_458.0

# Mean volumetric radius of the Earth (m).
R_EARTH = 6_371_000.0

# ITU standard atmosphere refraction factor (4/3 Earth model).
# Radio waves bend slightly downward in the troposphere.  The 4/3
# effective Earth radius model accounts for this so we can draw
# straight ray paths.  Source: ITU-R P.453.
K_REFR = 4.0 / 3.0

# Effective Earth radius (m) = (4/3) * 6,371,000 = 8,494,667 m.
R_EFF = K_REFR * R_EARTH

# Boltzmann constant (J/K). Exact by SI definition.
K_BOLTZ = 1.380649e-23


# ================================================================
#  KNICKEBEIN SYSTEM PARAMETERS
# ================================================================

# Operating frequency (Hz).  Knickebein used 30, 31.5, 33.3 MHz.
# We use 31.5 MHz as the primary frequency (confirmed by Bufton).
FREQ_DEFAULT = 31_500_000.0

# Wavelength (m) at 31.5 MHz.
LAM_DEFAULT = C / FREQ_DEFAULT

# Wavenumber (rad/m) at 31.5 MHz.
K_DEFAULT = 2.0 * np.pi / LAM_DEFAULT

# Transmit power (W). 3 kW rated output.
P_TX = 3000.0

# Antenna physical dimensions (m).
# 99m wide (horizontal), 29m tall (vertical).
# Source: Bauer (2004), p.12; Price (2017), p.24-25.
L_H = 99.0   # horizontal aperture
H_V = 29.0   # vertical aperture

# Physical aperture area (m^2).
A_PHYS = L_H * H_V

# Antenna directivity at 31.5 MHz.
# G = 4*pi*A / lambda^2.  Standard uniform aperture formula.
# Source: Balanis, Antenna Theory, Ch. 12.
G_DIR = 4.0 * np.pi * A_PHYS / LAM_DEFAULT**2
G_DIR_dB = 10.0 * np.log10(G_DIR)

# Knickebein squint angle (degrees). Each sub-beam is steered ±5°
# off the bore-sight by phasing its mast relative to the other,
# giving the ~500 yard equisignal corridor at operational range.
# Source: Trenkle (1979), p.67; Price (2017), p.24.
SQUINT_DEG = 5.0

# Equisignal crossover loss (dB), derived from aperture theory.
# A uniform aperture of width L_H has far-field amplitude pattern
#     F(θ) = sinc(π·L_H·sin(θ)/λ)   (sinc = sin(x)/x, unnormalised)
# so the loss relative to boresight at θ = SQUINT_DEG is
#     CROSSOVER_dB = 20·log10(|F(SQUINT_DEG)|)
# For L_H=99 m, λ≈9.517 m (31.5 MHz), squint=5° → −19.87 dB.
# Source: Balanis, Antenna Theory, Ch. 6 (uniform aperture).
_SQUINT_X = L_H * np.sin(np.radians(SQUINT_DEG)) / LAM_DEFAULT
CROSSOVER_dB = 20.0 * np.log10(abs(np.sinc(_SQUINT_X)))


# ================================================================
#  EQUISIGNAL CORRIDOR WIDTH (first-principles, sinc pattern)
# ================================================================
#
# Derived from the same sinc-squared aperture pattern that gives
# CROSSOVER_dB above, plus the pilot's auditory A/N discrimination
# threshold.  The pattern of each sub-beam is
#     F(θ) = sinc(L_H · sin(θ − θ_peak) / λ)
# with θ_peak = ±SQUINT_DEG for A-side / N-side beams.
#
# At boresight (θ_obs = 0) the two beams give identical amplitude
# (both are at squint angle from their own peak).  An aircraft that
# drifts by Δθ toward A sees its A-beam rise AND its N-beam fall, so
# the A/N imbalance in dB is
#     ΔL_A/N(Δθ)  ≈  2 · |dF/dθ|_{θ=SQUINT_DEG} · Δθ
#
# Clinical auditory discrimination studies (NATO AGARDograph 300
# Vol. 10, §6.2) converge on **1 dB** as the smallest A/N imbalance a
# trained pilot can hold on-beam by ear.  The half-width of the
# equisignal corridor in radians is therefore
#     Δθ_half  =  1 dB  /  [2 · |dF/dθ|_{θ=SQUINT_DEG}]
# and the full corridor width at slant range d is
#     W(d)  =  d · tan(2 · Δθ_half)  ≈  d · 2 · Δθ_half       (small angle)
#
# The analytic slope of F(θ) at θ = SQUINT_DEG is
#     dF/dθ = (20/ln 10) · [cot(u) − 1/u] · (π L_H cos θ / λ)
# with u = π · L_H · sin(θ) / λ.  At SQUINT_DEG = 5° it comes to
# roughly 1040 dB/rad, giving a corridor half-angle of ~0.028° and
# a corridor full width of ~460 yd at 439 km (Bufton's Spalding run,
# which he visually estimated at ~500 yd — within 7 % of first-
# principles prediction).

def sinc_pattern_slope_dB_per_rad(theta_deg=SQUINT_DEG,
                                   aperture_m=L_H,
                                   wavelen_m=LAM_DEFAULT):
    """
    Analytic slope of the uniform-aperture sinc pattern (in dB/rad)
    evaluated at angular offset ``theta_deg`` from boresight.

    F(θ) = 20·log10 |sinc(L·sin(θ)/λ)|     (normalised sinc, sin(u)/u)
    dF/dθ = (20/ln 10) · (π L cos θ / λ) · [cot(u) − 1/u]
    with u = π · L · sin(θ) / λ.
    """
    theta_rad = np.radians(theta_deg)
    u = np.pi * aperture_m * np.sin(theta_rad) / wavelen_m
    du_dtheta = np.pi * aperture_m * np.cos(theta_rad) / wavelen_m
    dlog_du = (np.cos(u) / np.sin(u)) - (1.0 / u)
    return (20.0 / np.log(10.0)) * du_dtheta * dlog_du


def equisignal_half_angle_rad(AN_threshold_dB=1.0,
                              theta_squint_deg=SQUINT_DEG,
                              aperture_m=L_H,
                              wavelen_m=LAM_DEFAULT):
    """
    Half-angular-width of the equisignal corridor (radians), i.e. the
    largest off-boresight angle at which the pilot hears ``AN_threshold_dB``
    of A/N imbalance.

    Uses the analytic sinc-pattern slope at the squint angle and the
    factor of 2 that accounts for one beam rising while the other falls.
    """
    slope = abs(sinc_pattern_slope_dB_per_rad(theta_deg=theta_squint_deg,
                                              aperture_m=aperture_m,
                                              wavelen_m=wavelen_m))
    return AN_threshold_dB / (2.0 * slope)


def equisignal_corridor_width_m(distance_m,
                                AN_threshold_dB=1.0,
                                theta_squint_deg=SQUINT_DEG,
                                aperture_m=L_H,
                                wavelen_m=LAM_DEFAULT):
    """
    Total on-ground width of the equisignal corridor at slant range
    ``distance_m``, in metres.

    The corridor is symmetric about the boresight, so the full width
    is 2·Δθ_half projected to range.
    """
    half = equisignal_half_angle_rad(
        AN_threshold_dB=AN_threshold_dB,
        theta_squint_deg=theta_squint_deg,
        aperture_m=aperture_m,
        wavelen_m=wavelen_m,
    )
    return distance_m * 2.0 * np.tan(half)


# Module-level convenience constant: canonical 1 dB pilot threshold
# half-angle in degrees (≈ 0.028°), used by downstream scripts that
# want a single "equisignal angular width" number like the DC_Dan
# 0.066° estimate in compute_equisignal_widths.py.
EQUISIGNAL_HALF_ANGLE_DEG = np.degrees(equisignal_half_angle_rad())


# --- Receiver noise environment (1940s era) ---

# Receiver noise figure (dB). 1940s vacuum tube receiver.
RX_NF_dB = 10.0

# Detection bandwidth (Hz). Matched-filter bandwidth for the Knickebein
# MCW / A2 emission: a 31.5 MHz carrier amplitude-modulated with a
# 1,150 Hz audio tone, keyed on/off at a few Hz to produce the Lorenz
# dot-dash pattern. Signal bandwidth is dominated by the audio tone
# and its keying sidebands, not by AM-voice bandwidth. 500 Hz gives a
# comfortable narrow-CW passband that fully admits the signal while
# rejecting the wideband noise that a 3 kHz AM-voice filter would let
# in. See Amendment E of the null doc for the noise-floor derivation
# and the bandwidth-choice reasoning.
RX_BW_Hz = 500.0

# System reference temperature (K). IEEE/ITU standard.
T_SYS = 290.0

# External noise figure (dB above kTB).  At 31.5 MHz, the dominant
# external noise source for an aircraft at altitude is galactic
# synchrotron radiation (cosmic radio background from electrons
# spiralling in the Milky Way's magnetic field).
#
# ITU-R P.372-16, Eq. 14 (galactic noise, curve E):
#   Fa = 52.0 - 23.0 * log10(f_MHz)
#   Fa = 52.0 - 23.0 * log10(31.5) = 52.0 - 34.5 = 17.5 dB
#
# This corresponds to a noise temperature of ~16,000 K (median).
# Range: ~5,000 K (galactic pole) to ~50,000 K (galactic centre).
#
# NOTE: Previous code used FA_dB = 32.0, which corresponds to
# "residential man-made noise" (ITU-R P.372, curve B).  This was
# incorrect for an aircraft at 6000m altitude over rural England
# at night in 1940.  Man-made noise from ground sources is heavily
# attenuated at aircraft altitude.  Galactic noise is the dominant
# contribution and is unavoidable (it comes from above).
#
# Source: ITU-R P.372-16, Table 1, Eq. 14 (galactic noise).
FA_dB = 18.0


# ================================================================
#  NOISE FLOOR (computed once, used everywhere)
# ================================================================

# Thermal noise power: N_thermal = k_B * T * B (in dBW)
N_THERMAL_dBW = 10.0 * np.log10(K_BOLTZ * T_SYS * RX_BW_Hz)

# Total noise floor: max(receiver NF, external noise) added to thermal
# At 31 MHz, galactic noise (32 dB) dominates over receiver NF (10 dB).
N_FLOOR_dBW = N_THERMAL_dBW + max(RX_NF_dB, FA_dB)


# ================================================================
#  USABILITY THRESHOLD
# ================================================================
#
# The operational criterion for the BotB null hypothesis is whether
# the RF signal is physically present at the receiver antenna above
# the galactic + thermal noise floor.  The FuBl 2 / EBL 3 airborne
# receiver has full AGC on its RF and first two IF stages, a manual
# sensitivity knob (Empfindlichkeitsregler), an automatic volume
# control, and a two-stage AF amplifier (D.(Luft) T.4058 Feb 1943,
# Section 24 "Schaltung des EBL 3").  Once RF signal is above the
# detection floor, the receiver electronics amplify it to audible
# SPL at the pilot's earphones; the cockpit acoustic environment is
# not a meaningful bottleneck.  Primary-source German bomber crew
# testimony (Bundesarchiv-MA RL 19-6/40 ref. 230Q18) confirms the
# Knickebein signals were "easily heard" in operational He 111
# cockpits, even through British Aspirin/Headache jamming.
#
# We therefore use:
#
#   DETECT_dB = +10 dB above the ITU-R P.372 galactic noise floor
#               (bare detection, standard radio-propagation threshold)
#
# Paths below 0 dB are DEAD (signal below thermal noise, no gain
# chain can amplify it).  Paths from 0 to +10 dB are MARGINAL.
# Paths >= +10 dB are USABLE.
#
# The previously-used +30 dB "operational audibility" threshold was
# based on the assumption that the EBL 3 was a non-AGC envelope
# detector where audio output is linear in RF input.  The primary
# source (D.(Luft) T.4058 Section 24) shows the receiver actually
# has full AGC + manual gain + ALC + 2-stage AF amp, so audio output
# is approximately constant across a wide RF input range.  The +30
# dB threshold has been removed; the receiver electronics do the
# work once the signal exists at the antenna.
DETECT_dB = 10.0
# Legacy alias names kept so older graph code still imports cleanly
RF_DETECT_dB = DETECT_dB
AF_AUDIBLE_dB = DETECT_dB


# ================================================================
#  ITU-R P.526-16 SMOOTH EARTH DIFFRACTION
# ================================================================
#
# The diffraction field strength relative to free space is:
#
#   20*log10(E/E0) = F(X) + G(Y1) + G(Y2)   dB
#
# where:
#   X = normalised path length
#   Y1 = normalised TX height
#   Y2 = normalised RX height
#   F(X) = distance attenuation term
#   G(Y) = height gain term
#
# Source: ITU-R P.526-16 (2025), Section 3.1.1.2, Eq. 13-18.
# ================================================================

# ----------------------------------------------------------------
#  Ground-type registry used by β / K computation below
# ----------------------------------------------------------------

# (sigma S/m, eps_r) pairs keyed by ground name.  These match the
# constants used by the grwave driver in make_p526_vs_p368_graphs.py
# so that the ITU P.526 §3 and the P.368 GRWAVE runs share a common
# electrical model of the ground.
GROUND_PARAMS = {
    "land": (5e-3, 15.0),   # average land, ITU-R P.527 class 3 (typical)
    "sea":  (5.0,  70.0),   # seawater, warm/typical North Sea values
    "wet":  (2e-2, 30.0),   # wet soil (rarely used)
    "dry":  (1e-3,  4.0),   # dry soil (rarely used)
}

# Effective-Earth-radius factor for 4/3 standard atmosphere.  Appears in
# ITU-R P.526 Eq. 16a as "k", the multiplier on the true Earth radius.
K_RADIUS_DEFAULT = 4.0 / 3.0


def p526_K_from_Eq16a(sigma, freq_Hz, k_radius=K_RADIUS_DEFAULT):
    """
    ITU-R P.526-16 Eq. 16a simplified K (disregards ε).

        K² ≈ 6.89·σ / (k^(2/3)·f^(5/3))

    where σ is in S/m, f in MHz, and k is the effective-Earth-radius
    multiplier.  Returns K (not K²).  This is the K used by Eq. 16 β
    and by the Eq. 18 G(Y) lower-bound clamp.
    """
    f_MHz = freq_Hz / 1e6
    K2 = 6.89 * sigma / (k_radius**(2.0/3.0) * f_MHz**(5.0/3.0))
    return np.sqrt(K2)


def p526_beta(sigma, freq_Hz, polarization="vertical", ground="land",
              k_radius=K_RADIUS_DEFAULT):
    """
    ITU-R P.526-16 Eq. 16 β parameter.

    ITU-R P.526-16 §3.1.1.2 rule (verbatim from the Recommendation):
      - Horizontal polarisation at all frequencies: β = 1.
      - Vertical polarisation above 20 MHz over land: β = 1.
      - Vertical polarisation above 300 MHz over sea: β = 1.
      - Otherwise (vertical pol below those cuts): β must be computed
        from K via Eq. 16.

    Eq. 16:
        β = (1 + 1.6·K² + 0.67·K⁴) / (1 + 4.5·K² + 1.53·K⁴)

    Returns the tuple (β, K) so downstream code can also apply the
    Eq. 18 G(Y) floor at 2 + 20·log10(K).  If β=1 because of the
    frequency rule, K is returned as 0.0 (no clamp applied).

    Parameters
    ----------
    sigma        : ground conductivity (S/m)
    freq_Hz      : frequency (Hz)
    polarization : "vertical" (default for ground-wave) or "horizontal"
    ground       : "land" or "sea" — selects the 20/300 MHz cut
    k_radius     : effective-Earth-radius factor (default 4/3)
    """
    f_MHz = freq_Hz / 1e6
    # Horizontal polarisation: β = 1 at all frequencies
    if polarization == "horizontal":
        return 1.0, 0.0
    # Vertical polarisation: frequency cut depends on ground type
    cut_MHz = 20.0 if ground == "land" else 300.0
    if f_MHz > cut_MHz:
        return 1.0, 0.0
    # Otherwise compute via Eq. 16 + 16a
    K = p526_K_from_Eq16a(sigma, freq_Hz, k_radius=k_radius)
    K2 = K * K
    K4 = K2 * K2
    numerator   = 1.0 + 1.6 * K2 + 0.67 * K4
    denominator = 1.0 + 4.5 * K2 + 1.53 * K4
    return numerator / denominator, K


def itu_normalised_distance(d, lam, a_e=R_EFF, beta=1.0):
    """
    ITU-R P.526 normalised path length X.

    Parameters
    ----------
    d     : great-circle distance (m)
    lam   : wavelength (m)
    a_e   : effective Earth radius (m)
    beta  : terrain/polarisation parameter (Eq. 16; 1.0 when the ITU
            frequency-cut rule says so, otherwise from p526_beta())

    Source: ITU-R P.526-16, Eq. 14.
    """
    return beta * (np.pi / (lam * a_e**2))**(1.0/3.0) * d


def itu_normalised_height(h, lam, a_e=R_EFF, beta=1.0):
    """
    ITU-R P.526 normalised antenna height Y.

    Parameters
    ----------
    h     : antenna height above surface (m)
    lam   : wavelength (m)
    a_e   : effective Earth radius (m)
    beta  : terrain/polarisation parameter

    Source: ITU-R P.526-16, Eq. 14.
    """
    return 2.0 * beta * (np.pi**2 / (lam**2 * a_e))**(1.0/3.0) * h


def itu_distance_term(X):
    """
    ITU-R P.526 distance attenuation F(X) in dB.

    For X >= 1.6:  F(X) = 11 + 10*log10(X) - 17.6*X
    For X <  1.6:  F(X) = -20*log10(X) - 5.6488*X^1.425

    Source: ITU-R P.526-16, Eq. 15-16.
    """
    if X >= 1.6:
        return 11.0 + 10.0 * np.log10(X) - 17.6 * X
    else:
        return -20.0 * np.log10(max(X, 1e-10)) - 5.6488 * X**1.425


def itu_height_gain(Y, beta=1.0, K=None):
    """
    ITU-R P.526 height gain G(Y) in dB.

    B = beta * Y
    For B >  2:  G = 17.6*(B-1.1)^0.5 - 5*log10(B-1.1) - 8     (Eq. 18)
    For B <= 2:  G = 20*log10(B + 0.1*B^3)                     (Eq. 18a)

    Eq. 18 lower bound (from ITU-R P.526-16 §3.1.1.2, page 10):
        "If G(Y) < 2 + 20·log10(K), set G(Y) to the value 2 + 20·log10(K)."
    This clamp is applied when K is provided and K > 0.  For β=1 cases
    (horizontal pol, or vert pol above the frequency cut) the clamp does
    not apply and K should be passed as None or 0.

    Source: ITU-R P.526-16, Eq. 18.
    """
    B = beta * Y
    if B > 2.0:
        G = 17.6 * np.sqrt(B - 1.1) - 5.0 * np.log10(B - 1.1) - 8.0
    else:
        # Guard against B=0 (antenna on surface)
        B_eff = max(B, 1e-10)
        G = 20.0 * np.log10(B_eff + 0.1 * B_eff**3)

    # Apply Eq. 18 lower-bound clamp when K is provided (vert pol below
    # the frequency cut).  For K <= 0 or K None the clamp is skipped.
    if K is not None and K > 0.0:
        G_floor = 2.0 + 20.0 * np.log10(K)
        if G < G_floor:
            G = G_floor
    return G


def itu_diffraction_loss(d, h_tx, h_rx, freq=FREQ_DEFAULT, a_e=R_EFF,
                          ground="land", polarization="vertical"):
    """
    Total smooth Earth diffraction loss using ITU-R P.526-16.

    Returns the diffraction loss in dB (positive = signal weaker
    than free space).  Combines the distance term F(X) and height
    gains G(Y1), G(Y2) per Eq. 13-18.

    Parameters
    ----------
    d           : great-circle distance (m)
    h_tx        : transmitter height above surface (m)
    h_rx        : receiver height above surface (m)
    freq        : frequency (Hz)
    a_e         : effective Earth radius (m)
    ground      : "land" or "sea" (default "land"). Selects the
                  conductivity and the ITU 20/300 MHz β cut. For
                  pure overland paths use "land"; for overwater paths
                  (Stollberg, Greny, Telefunken) use "sea".
    polarization: "vertical" (default; Knickebein is vertical) or
                  "horizontal"

    β (Eq. 16) and K (Eq. 16a) are computed automatically per the
    ITU rule: if horizontal pol, or vert pol above 20 MHz over land,
    or vert pol above 300 MHz over sea, β=1 and the G(Y) clamp is
    not applied. Otherwise β is calculated from K and the Eq. 18
    G(Y) lower bound 2 + 20·log10(K) is enforced.

    Returns
    -------
    dict with:
        loss_dB   : total diffraction loss (dB, positive)
        F_X_dB    : distance term F(X) (dB, negative)
        G_Y1_dB   : TX height gain (dB, can be negative or positive)
        G_Y2_dB   : RX height gain (dB, can be negative or positive)
        X         : normalised distance
        Y1        : normalised TX height
        Y2        : normalised RX height
        beta      : the β used (1.0 or from Eq. 16)
        K         : the K used for the clamp (0.0 when β=1)
    """
    lam = C / freq

    # β and K per ITU-R P.526-16 Eq. 16 + 16a.  For land at ≥31.5 MHz
    # vert pol this gives β=1 (above the 20 MHz cut).  For sea at
    # 31.5 MHz vert pol this gives β≈0.81 and K≈0.30 (below 300 MHz).
    sigma = GROUND_PARAMS[ground][0]
    beta, K = p526_beta(sigma, freq,
                        polarization=polarization, ground=ground)
    K_clamp = K if K > 0.0 else None

    X  = itu_normalised_distance(d, lam, a_e, beta=beta)
    Y1 = itu_normalised_height(h_tx, lam, a_e, beta=beta)
    Y2 = itu_normalised_height(h_rx, lam, a_e, beta=beta)

    F_X  = itu_distance_term(X)
    G_Y1 = itu_height_gain(Y1, beta=beta, K=K_clamp)
    G_Y2 = itu_height_gain(Y2, beta=beta, K=K_clamp)

    # E/E0 (dB) = F(X) + G(Y1) + G(Y2)
    # F(X) is large negative (signal decays with distance)
    # G(Y) can be positive or negative
    total_dB = F_X + G_Y1 + G_Y2

    # Loss is the negative of total (positive = signal weaker)
    loss_dB = -total_dB

    # IMPORTANT: The ITU-R P.526 smooth Earth diffraction formula is
    # only valid in the diffraction region (beyond the radio horizon).
    # Within line of sight there is no diffraction obstruction, so
    # both flat and globe models give the same result (FSPL only).
    #
    # Rule 1: Within LoS (d < d_los), loss = 0.  No exceptions.
    # Rule 2: Beyond LoS, use ITU formula but cap at 0 minimum
    #         (the globe can only make things worse, never better).
    d_los = np.sqrt(2.0 * a_e * h_tx) + np.sqrt(2.0 * a_e * h_rx)
    if d < d_los:
        loss_dB = 0.0
    elif loss_dB < 0.0:
        loss_dB = 0.0

    return dict(
        loss_dB=loss_dB,
        F_X_dB=F_X,
        G_Y1_dB=G_Y1,
        G_Y2_dB=G_Y2,
        X=X, Y1=Y1, Y2=Y2,
        beta=beta, K=K,
    )


# ================================================================
#  SOMMERFELD-NORTON PLANE-EARTH GROUND WAVE
# ================================================================
#
# Plane finitely-conducting earth ground-wave field strength per
# ITU Handbook on Ground Wave Propagation (2014 edition), Part 1
# §3.2.1, equations (3), (5), (6), (7), (8).  Three-term coherent
# sum of direct ray, Fresnel-reflected ray, and surface-wave
# (Norton 1937 attenuation function F).
#
# This is the rigorous flat-Earth solution.  For elevated VHF
# geometries the surface-wave F term is negligible and the result
# collapses to a coherent two-ray model, producing Fresnel-zone
# multipath lobes at short range that fade into the Friis envelope
# past about 300 km.  See /tmp/sommerfeld_norton_check.py for the
# original derivation and a standalone sweep verification.
#
# We include this in the BotB analysis because P.368-10 NOTE 1
# explicitly references it as the rigorous flat-Earth method, and
# because it sits between Friis flat-Earth (single ray) and
# P.526/P.368 globe-diffraction (residue series) in the hierarchy
# of propagation models.
# ================================================================

ETA0 = 119.9169832 * np.pi    # intrinsic impedance of free space (Ω)


def _sn_dipole_moment_for_power(P_W, freq=FREQ_DEFAULT):
    """
    Short vertical Hertzian dipole: P = (η₀/(12π))·(k·I·dl)².
    Solve for the dipole moment I·dl (A·m) that radiates P_W watts.
    """
    k = 2.0 * np.pi * freq / C
    return np.sqrt(12.0 * np.pi * P_W / ETA0) / k


def sommerfeld_norton_Ez(d_m, h_tx_m, h_rx_m, freq, sigma, eps_r,
                         I_dl):
    """
    Vertical-component E field at range d_m over a lossy plane Earth,
    for a short vertical current element at (0, h_tx_m) radiating to
    an observer at (d_m, h_rx_m).

    Parameters
    ----------
    d_m    : horizontal range in metres
    h_tx_m : TX height in metres
    h_rx_m : RX height in metres
    freq   : Hz
    sigma  : ground conductivity (S/m)
    eps_r  : ground relative permittivity
    I_dl   : dipole moment of a short vertical current element (A·m)

    Returns
    -------
    complex E_z in V/m.  Take abs() for magnitude.

    Source: ITU Handbook on Ground Wave Propagation (2014), Part 1
    §3.2.1, Eq. 3 (three-term vertical component) + 5-8 (attenuation
    function and numerical distance).  Theory: Sommerfeld (1909) +
    Norton (1936, 1937, 1941).
    """
    from scipy.special import wofz

    lam = C / freq
    k   = 2.0 * np.pi / lam
    f_MHz = freq / 1e6

    # Ground electrical parameter x = σ/(ω·ε₀)    (Handbook Eq. 8)
    x = 1.8e4 * sigma / f_MHz

    # u² = 2/(ε_r − jx)    (Handbook Eq. 7)
    u2 = 2.0 / (eps_r - 1j * x)
    u4 = u2 * u2

    # Geometry: direct ray r1 and image-reflected ray r2.
    dh1 = h_rx_m - h_tx_m
    dh2 = h_rx_m + h_tx_m
    r1  = np.sqrt(d_m ** 2 + dh1 ** 2)
    r2  = np.sqrt(d_m ** 2 + dh2 ** 2)
    sin_psi1 = dh1 / r1
    cos_psi1 = d_m / r1
    sin_psi2 = dh2 / r2
    cos_psi2 = d_m / r2
    _ = sin_psi1   # explicit: used implicitly via cos² in Eq. 3

    # Fresnel reflection coefficient for vertical polarisation
    n2 = eps_r - 1j * x
    root = np.sqrt(n2 - cos_psi2 ** 2)
    Rv   = (n2 * sin_psi2 - root) / (n2 * sin_psi2 + root)

    # Numerical distance w    (Handbook Eq. 6)
    w = (-1j * 2.0 * k * r2 * u2 * (1.0 - u2 * cos_psi2 ** 2)) / (1.0 - Rv)

    # Attenuation function F (Handbook Eq. 5). For |w|>10 the
    # exp(-w)·erfc(...) form overflows; use the large-argument
    # asymptotic series instead.
    if np.abs(w) > 10.0:
        F = (-1.0 / (2.0 * w)
             - 3.0 / (2.0 * w) ** 2
             - 15.0 / (2.0 * w) ** 3
             - 105.0 / (2.0 * w) ** 4)
    else:
        sqrt_w = np.sqrt(w)
        # wofz(z) = exp(-z²)·erfc(-jz), so exp(-w)·erfc(-j√w) = wofz(√w).
        F = 1.0 - 1j * np.sqrt(np.pi * w) * wofz(sqrt_w)

    direct    = cos_psi1 ** 2 * np.exp(-1j * k * r1) / r1
    reflected = cos_psi2 ** 2 * Rv * np.exp(-1j * k * r2) / r2
    surface   = ((1.0 - Rv) * (1.0 - u2 + u4 * cos_psi2 ** 2)
                 * F * np.exp(-1j * k * r2) / r2)

    Ez = 1j * 30.0 * k * I_dl * (direct + reflected + surface)
    return Ez


def sommerfeld_norton_snr_peak(d_km, h_tx_m, h_rx_m, ground="sea",
                                 freq=FREQ_DEFAULT, rx_gain_dBi=0.0):
    """
    Sommerfeld-Norton plane-Earth peak SNR above the BotB noise floor.

    Computes |E_z| from the rigorous flat-Earth three-term formula,
    applies the Knickebein 26 dBi directional TX gain as a field-
    strength boost on top of the short-dipole baseline, and converts
    to P_rx at an isotropic-equivalent receiver with the given
    rx_gain_dBi.

    The short-dipole directivity D = 1.5 (≈1.76 dBi); the
    directional-array gain enters as sqrt(G_dir_linear / 1.5) on the
    field-strength side.  This gives the same EIRP-equivalent as the
    Knickebein 3 kW / 26 dBi Wellenspiegel array in broadside.
    """
    I_dl = _sn_dipole_moment_for_power(P_TX, freq=freq)
    sigma, eps_r = GROUND_PARAMS[ground]
    Ez = sommerfeld_norton_Ez(d_km * 1000.0, h_tx_m, h_rx_m,
                               freq, sigma, eps_r, I_dl)
    SHORT_DIP_D = 1.5
    extra_gain_lin = (10 ** (G_DIR_dB / 10.0)) / SHORT_DIP_D
    E_boosted = np.abs(Ez) * np.sqrt(extra_gain_lin)
    # Convert |E| (V/m) to P_rx via isotropic effective aperture.
    # |Ez| from sommerfeld_norton_Ez is a PEAK-amplitude phasor
    # (follows from the j·30·k·I·dl normalisation, where I·dl is peak).
    # Time-averaged Poynting uses <E(t)²> = |E_peak|²/2, so the
    # conversion to P_rx carries an explicit factor of 1/2:
    #     P_rx = (|E_peak|²/(2·η₀))·(λ²/(4π))·G_rx_lin
    # Skipping the 1/2 adds a spurious +3 dB to every SN result
    # (verified in free-space limit: SN should equal Friis but
    # overshot by 3.5 dB before the fix was applied).
    lam = C / freq
    G_rx_lin = 10 ** (rx_gain_dBi / 10.0)
    A_eff = (lam ** 2 / (4.0 * np.pi)) * G_rx_lin
    P_rx_W = (E_boosted ** 2 / (2.0 * ETA0)) * A_eff
    if P_rx_W <= 0:
        return -np.inf
    P_rx_dBW = 10.0 * np.log10(P_rx_W)
    return P_rx_dBW - N_FLOOR_dBW


# ================================================================
#  LINE OF SIGHT GEOMETRY
# ================================================================

def line_of_sight(h_tx, h_rx, a_e=R_EFF):
    """
    Radio line-of-sight range for given TX and RX heights.

    d_los = sqrt(2*a_e*h_tx) + sqrt(2*a_e*h_rx)

    Source: ITU-R P.1546; standard radio horizon formula.
    """
    d_tx = np.sqrt(2.0 * a_e * h_tx)
    d_rx = np.sqrt(2.0 * a_e * h_rx)
    return d_tx + d_rx


# ================================================================
#  LINK BUDGET
# ================================================================

def link_budget(d, diff_loss_dB, freq=FREQ_DEFAULT, rx_gain_dBi=0.0):
    """
    Compute received signal power and SNR.

    P_rx = P_tx + G_tx + G_rx - FSPL - diff_loss
    SNR  = P_rx - N_floor

    Parameters
    ----------
    d             : distance (m)
    diff_loss_dB  : diffraction loss (dB, positive). 0 for flat model.
    freq          : frequency (Hz)
    rx_gain_dBi   : receiver antenna gain (dBi)

    Returns
    -------
    dict with P_rx_dBW, SNR_dB, FSPL_dB
    """
    lam = C / freq
    FSPL = 20.0 * np.log10(4.0 * np.pi * d / lam)
    P_rx = (10.0 * np.log10(P_TX) + G_DIR_dB + rx_gain_dBi
            - FSPL - diff_loss_dB)
    SNR = P_rx - N_FLOOR_dBW
    return dict(P_rx_dBW=P_rx, SNR_dB=SNR, FSPL_dB=FSPL)


# ================================================================
#  LOAD PATH DATA FROM CSV
# ================================================================

def load_paths(csv_path=None, csv_name="knickebein_paths.csv"):
    """
    Load a path dataset CSV.

    Parameters
    ----------
    csv_path : str, optional
        Full path to CSV file. If None, uses csv_name in script directory.
    csv_name : str
        Default CSV file name (knickebein_paths.csv, xgerat_paths.csv, etc.)

    Returns a list of dicts, one per path.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), csv_name)

    paths = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(
            (row for row in f if not row.startswith("#")),
        )
        for row in reader:
            # ground_type is optional for backwards compat.  If missing
            # or blank it defaults to "land", which keeps β=1 at VHF
            # (ITU P.526-16 Eq. 16 allows β=1 for vert pol above 20 MHz
            # over land).  Overwater paths must set ground_type = "sea".
            g_raw = row.get("ground_type", "") or ""
            ground_type = g_raw.strip().lower() or "land"
            paths.append({
                "path_id":    row["path_id"].strip(),
                "type":       row["type"].strip(),
                "target":     row["target"].strip(),
                "tx_station": row["tx_station"].strip(),
                "tx_alt_m":   float(row["tx_alt_m"]),
                "rx_alt_m":   float(row["rx_alt_m"]),
                "freq_mhz":   float(row["freq_mhz"]),
                "rx_gain_dbi": float(row["rx_gain_dbi"]),
                "distance_m": float(row["distance_m"]),
                "source":     row["source"].strip(),
                "ground_type": ground_type,
            })
    return paths


# ================================================================
#  MAIN ANALYSIS
# ================================================================

def analyse_all_paths(csv_name="knickebein_paths.csv"):
    """
    Run full analysis on every path in the specified CSV.

    For each path, compute:
      - Line-of-sight range
      - ITU-R P.526 diffraction loss (globe model)
      - Flat model SNR (FSPL only, no diffraction)
      - Globe model SNR (FSPL + ITU diffraction)
      - Equisignal SNR (peak - 19 dB crossover)
    """
    paths = load_paths(csv_name=csv_name)
    results = []

    for p in paths:
        d       = p["distance_m"]
        h_tx    = p["tx_alt_m"]
        h_rx    = p["rx_alt_m"]
        freq    = p["freq_mhz"] * 1e6
        rx_gain = p["rx_gain_dbi"]
        ground  = p.get("ground_type", "land")

        # Line of sight
        d_los = line_of_sight(h_tx, h_rx)
        d_shadow = max(0, d - d_los)

        # ITU diffraction (globe model) — β and Eq. 18 clamp picked up
        # automatically from the ground type (land/sea).
        itu = itu_diffraction_loss(d, h_tx, h_rx, freq, ground=ground)

        # Link budgets
        flat  = link_budget(d, 0.0, freq, rx_gain)
        globe = link_budget(d, itu["loss_dB"], freq, rx_gain)

        results.append({
            **p,
            "d_los_km":       d_los / 1000.0,
            "d_shadow_km":    d_shadow / 1000.0,
            "itu_loss_dB":    itu["loss_dB"],
            "itu_F_X_dB":     itu["F_X_dB"],
            "itu_G_Y1_dB":    itu["G_Y1_dB"],
            "itu_G_Y2_dB":    itu["G_Y2_dB"],
            "itu_X":          itu["X"],
            "itu_Y1":         itu["Y1"],
            "itu_Y2":         itu["Y2"],
            "flat_SNR_peak":  flat["SNR_dB"],
            "flat_SNR_eq":    flat["SNR_dB"] + CROSSOVER_dB,
            "globe_SNR_peak": globe["SNR_dB"],
            "globe_SNR_eq":   globe["SNR_dB"] + CROSSOVER_dB,
            "FSPL_dB":        flat["FSPL_dB"],
        })

    return results


def print_results(results):
    """Pretty-print the analysis results."""
    SEP = "=" * 90

    print(SEP)
    print("  BATTLE OF THE BEAMS — ITU-R P.526-16 PROPAGATION ANALYSIS")
    print("  Height gain: ITU-R P.526-16 (2025), Eq. 13-18")
    print(f"  Noise floor: {N_FLOOR_dBW:.1f} dBW")
    print(f"  TX power: {P_TX:.0f} W ({10*np.log10(P_TX):.1f} dBW)")
    print(f"  TX gain: {G_DIR_dB:.1f} dBi (aperture {L_H}m x {H_V}m)")
    print(f"  Equisignal crossover: {CROSSOVER_dB:.2f} dB ({SQUINT_DEG:.0f} deg squint, ~500 yd)")
    print(f"  Detection floor: +{DETECT_dB:.0f} dB above noise (AGC/ALC delivers audible tone above this)")
    print(SEP)

    # Group by type
    for path_type, label in [
        ("operational", "CONFIRMED OPERATIONAL PATHS"),
        ("measurement", "BEAM MEASUREMENT / DETECTION"),
        ("intercept",   "ENIGMA INTERCEPTS"),
        ("telefunken",  "TELEFUNKEN TEST RANGES (4000m altitude)"),
    ]:
        group = [r for r in results if r["type"] == path_type]
        if not group:
            continue

        print(f"\n  {label}")
        print(f"  {'-' * 86}")
        print(f"  {'Path':<22} {'Dist':>6} {'LoS':>6} {'Shadow':>6} "
              f"{'ITU loss':>8} {'Flat pk':>8} {'Globe pk':>9} "
              f"{'Globe eq':>9} {'Status':>10}")
        print(f"  {'':22} {'km':>6} {'km':>6} {'km':>6} "
              f"{'dB':>8} {'dB':>8} {'dB':>9} {'dB':>9}")
        print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} "
              f"{'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*10}")

        for r in group:
            d_km = r["distance_m"] / 1000.0
            # Three-way classification against the receiver noise floor:
            #   USABLE   >= +10 dB (clearly above noise, AGC chain delivers
            #                      audible tone)
            #   MARGINAL  0 to +10 dB (at or near the noise floor)
            #   DEAD     < 0 dB (below thermal + galactic noise, no gain
            #                    chain can recover signal)
            if r["globe_SNR_eq"] >= DETECT_dB:
                status = "USABLE"
            elif r["globe_SNR_eq"] >= 0:
                status = "MARGINAL"
            else:
                status = "DEAD"

            name = f"{r['tx_station'][:8]}>{r['target'][:12]}"
            print(f"  {name:<22} {d_km:>6.0f} {r['d_los_km']:>6.0f} "
                  f"{r['d_shadow_km']:>6.0f} {r['itu_loss_dB']:>8.1f} "
                  f"{r['flat_SNR_peak']:>8.1f} {r['globe_SNR_peak']:>9.1f} "
                  f"{r['globe_SNR_eq']:>9.1f} {status:>10}")

    # Summary
    print(f"\n  {'=' * 86}")
    print("  SUMMARY")
    print(f"  {'=' * 86}")

    usable   = [r for r in results if r["globe_SNR_eq"] >= DETECT_dB]
    marginal = [r for r in results if 0 <= r["globe_SNR_eq"] < DETECT_dB]
    dead     = [r for r in results if r["globe_SNR_eq"] < 0]

    print(f"\n  Globe equisignal USABLE (>= {DETECT_dB:.0f} dB above noise floor): "
          f"{len(usable)} paths")
    for r in usable:
        margin = r["globe_SNR_eq"] - DETECT_dB
        print(f"    {r['tx_station']} > {r['target']}: "
              f"{r['distance_m']/1000:.0f} km, "
              f"globe eq SNR = {r['globe_SNR_eq']:.1f} dB "
              f"(+{margin:.1f} dB over detection floor)")

    print(f"\n  Globe equisignal MARGINAL (0 to {DETECT_dB:.0f} dB, "
          f"at the noise floor): {len(marginal)} paths")
    for r in marginal:
        print(f"    {r['tx_station']} > {r['target']}: "
              f"{r['distance_m']/1000:.0f} km, "
              f"globe eq SNR = {r['globe_SNR_eq']:.1f} dB")

    print(f"\n  Globe equisignal DEAD (< 0 dB, below noise floor): "
          f"{len(dead)} paths")
    for r in dead:
        print(f"    {r['tx_station']} > {r['target']}: "
              f"{r['distance_m']/1000:.0f} km, "
              f"globe eq SNR = {r['globe_SNR_eq']:.1f} dB")

    # Telefunken breakdown
    tf_paths = [r for r in results if r["type"] == "telefunken"]
    if tf_paths:
        print(f"\n  TELEFUNKEN RANGE ANALYSIS (4000m altitude, ITU height gain)")
        print(f"  Y_rx at 4000m = {tf_paths[0]['itu_Y2']:.2f}, "
              f"ITU G(Y_rx) = {tf_paths[0]['itu_G_Y2_dB']:.1f} dB")
        tf_dead = [r for r in tf_paths if r["globe_SNR_eq"] < DETECT_dB]
        tf_ok   = [r for r in tf_paths if r["globe_SNR_eq"] >= DETECT_dB]
        print(f"  Globe reaches detection floor ({DETECT_dB:.0f} dB) at: "
              f"{len(tf_ok)}/{len(tf_paths)} configs")
        print(f"  Globe FAILS detection floor at: "
              f"{len(tf_dead)}/{len(tf_paths)} configs")


# ================================================================
#  GRAPH GENERATION
# ================================================================

def generate_per_path_graphs(results, outdir=None, prefix="itu"):
    """
    Generate individual SNR vs distance graphs for each beam pairing.

    Each graph shows:
      - Flat model SNR (green line)
      - Globe model SNR with ITU P.526 (red line)
      - Noise floor (white solid, thin)
      - Detection floor +10 dB (faint yellow dashed)
      - The actual path distance marked
    """
    if outdir is None:
        outdir = os.path.dirname(__file__)

    plt.style.use('dark_background')

    # Only graph operational and Telefunken paths
    graph_paths = [r for r in results
                   if r["type"] in ("operational", "measurement", "telefunken", "intercept", "trial")]

    for r in graph_paths:
        fig, ax = plt.subplots(figsize=(12, 6))

        d_actual = r["distance_m"] / 1000.0
        h_tx = r["tx_alt_m"]
        h_rx = r["rx_alt_m"]
        freq = r["freq_mhz"] * 1e6
        rx_gain = r["rx_gain_dbi"]
        ground = r.get("ground_type", "land")

        # Compute SNR curves from 50 km to max(d+200, 900) km
        # For long paths (>=1000 km), extend d_max further so the
        # legend in upper right has room away from the intersection
        # points near the middle of the graph.
        if d_actual >= 1000:
            d_max = d_actual + 400
            d_min_plot = 10   # start curves near 0 to fill the axis
        else:
            d_max = max(d_actual + 200, 900)
            d_min_plot = 50
        distances_km = np.arange(d_min_plot, d_max + 1, 5)

        flat_snr = []
        globe_snr = []

        for dk in distances_km:
            dm = dk * 1000.0
            itu_r = itu_diffraction_loss(dm, h_tx, h_rx, freq, ground=ground)
            f = link_budget(dm, 0.0, freq, rx_gain)
            g = link_budget(dm, itu_r["loss_dB"], freq, rx_gain)
            flat_snr.append(f["SNR_dB"])
            globe_snr.append(g["SNR_dB"])

        flat_snr = np.array(flat_snr)
        globe_snr = np.array(globe_snr)

        # Computed values at the actual path distance for legend labels
        flat_pk  = r["flat_SNR_peak"]
        flat_eq  = r["flat_SNR_eq"]
        globe_pk = r["globe_SNR_peak"]
        globe_eq = r["globe_SNR_eq"]

        # Plot with computed values in the legend (zorder=3 to sit above vertical reference lines)
        ax.plot(distances_km, flat_snr, color='#4CAF50', linewidth=2.5,
                label=f'Flat peak: {flat_pk:.1f} dB', linestyle='-', zorder=3)
        ax.plot(distances_km, globe_snr, color='#FF1493', linewidth=2.5,
                label=f'Globe peak: {globe_pk:.1f} dB', linestyle='-', zorder=3)

        # Equisignal lines
        ax.plot(distances_km, flat_snr + CROSSOVER_dB, color='#4CAF50',
                linewidth=1.5, linestyle='--', alpha=0.6, zorder=3,
                label=f'Flat equisignal: {flat_eq:.1f} dB')
        ax.plot(distances_km, globe_snr + CROSSOVER_dB, color='#FF1493',
                linewidth=1.5, linestyle='--', alpha=0.6, zorder=3,
                label=f'Globe equisignal: {globe_eq:.1f} dB')

        # Noise floor (0 dB SNR): white solid, thin
        ax.axhline(y=0, color='white', linewidth=1.0, linestyle='-',
                   label='Noise floor', zorder=5)
        # Detection floor (10 dB): faint yellow dashed
        ax.axhline(y=DETECT_dB, color='#FFEB3B', linewidth=1.4,
                   linestyle='--', alpha=0.75,
                   label=f'Detection floor ({DETECT_dB:.0f} dB above noise)',
                   zorder=5)

        # Mark actual path distance (target) - in front of signal curves
        ax.axvline(x=d_actual, color='#90C4E0', linewidth=2, linestyle='-',
                   alpha=0.9, zorder=6,
                   label=f'Target: {d_actual:.0f} km')

        # Mark Radio Horizon (same blue, dashed) - behind signal curves
        d_los = line_of_sight(h_tx, h_rx) / 1000.0
        ax.axvline(x=d_los, color='#90C4E0', linewidth=1.5,
                   linestyle=(0, (9, 4)), alpha=0.9, zorder=1,
                   label=f'Radio Horizon: {d_los:.0f} km')

        # Shade below audibility threshold (faint red)
        ax.axhspan(-300, 0, alpha=0.06, color='#FFD54F', zorder=0)

        # Labels
        title = f"{r['tx_station']} > {r['target']}"
        subtitle = (f"{r['distance_m']/1000:.0f} km | "
                    f"TX {h_tx:.0f}m | RX {h_rx:.0f}m | "
                    f"{r['freq_mhz']:.1f} MHz | "
                    f"Squint {SQUINT_DEG:.0f}° ({CROSSOVER_dB:.2f} dB) | "
                    f"Globe eq SNR = {r['globe_SNR_eq']:.1f} dB")

        ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('SNR (dB)', fontsize=12)
        # For very long paths (>=1000 km), start x-axis at 0 so the
        # legend has room in the upper right without colliding with
        # the intersection points near the middle of the graph.
        x_start = 0 if d_actual >= 1000 else 50
        ax.set_xlim(x_start, d_max)
        ax.set_ylim(-150, 150)
        ax.grid(alpha=0.2, color='gray')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.3)

        fig.tight_layout()

        fname = f"{prefix}_{r['path_id'].lower()}.png"
        outpath = os.path.join(outdir, fname)
        plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        print(f"  Saved: {fname}")


def generate_per_path_graphs_equisignal_only(results, outdir=None, prefix="eq"):
    """
    Generate simplified equisignal-only graphs in dB.

    Same as generate_per_path_graphs but only shows the equisignal
    curves (not the beam peak).  Cleaner visualization focused on
    what matters for dot/dash discrimination: the crossover signal
    level that the pilot actually hears.
    """
    if outdir is None:
        outdir = os.path.dirname(__file__)

    plt.style.use('dark_background')

    graph_paths = [r for r in results
                   if r["type"] in ("operational", "measurement", "telefunken", "intercept", "trial")]

    for r in graph_paths:
        fig, ax = plt.subplots(figsize=(12, 6))

        d_actual = r["distance_m"] / 1000.0
        h_tx = r["tx_alt_m"]
        h_rx = r["rx_alt_m"]
        freq = r["freq_mhz"] * 1e6
        rx_gain = r["rx_gain_dbi"]
        ground = r.get("ground_type", "land")

        # For long paths (>=1000 km), extend d_max further so the
        # legend in upper right has room away from the intersection
        # points near the middle of the graph.
        if d_actual >= 1000:
            d_max = d_actual + 400
            d_min_plot = 10   # start curves near 0 to fill the axis
        else:
            d_max = max(d_actual + 200, 900)
            d_min_plot = 50
        distances_km = np.arange(d_min_plot, d_max + 1, 5)

        flat_snr = []
        globe_snr = []

        for dk in distances_km:
            dm = dk * 1000.0
            itu_r = itu_diffraction_loss(dm, h_tx, h_rx, freq, ground=ground)
            f = link_budget(dm, 0.0, freq, rx_gain)
            g = link_budget(dm, itu_r["loss_dB"], freq, rx_gain)
            flat_snr.append(f["SNR_dB"])
            globe_snr.append(g["SNR_dB"])

        flat_snr = np.array(flat_snr)
        globe_snr = np.array(globe_snr)

        # Equisignal values at the actual path distance
        flat_eq = r["flat_SNR_eq"]
        globe_eq = r["globe_SNR_eq"]

        # Plot ONLY the equisignal lines (not the beam peaks), zorder=3
        ax.plot(distances_km, flat_snr + CROSSOVER_dB, color='#4CAF50',
                linewidth=2.5, linestyle='-', zorder=3,
                label=f'Flat equisignal: {flat_eq:.1f} dB')
        ax.plot(distances_km, globe_snr + CROSSOVER_dB, color='#FF1493',
                linewidth=2.5, linestyle='-', zorder=3,
                label=f'Globe equisignal: {globe_eq:.1f} dB')

        # Noise floor (0 dB SNR): white solid, thin
        ax.axhline(y=0, color='white', linewidth=1.0, linestyle='-',
                   label='Noise floor', zorder=5)
        # Detection floor (10 dB): faint yellow dashed
        ax.axhline(y=DETECT_dB, color='#FFEB3B', linewidth=1.4,
                   linestyle='--', alpha=0.75,
                   label=f'Detection floor ({DETECT_dB:.0f} dB above noise)',
                   zorder=5)

        # Mark actual path distance (target) - in front of signal curves
        ax.axvline(x=d_actual, color='#90C4E0', linewidth=2, linestyle='-',
                   alpha=0.9, zorder=6,
                   label=f'Target: {d_actual:.0f} km')

        # Mark Radio Horizon (same blue, dashed) - behind signal curves
        d_los = line_of_sight(h_tx, h_rx) / 1000.0
        ax.axvline(x=d_los, color='#90C4E0', linewidth=1.5,
                   linestyle=(0, (9, 4)), alpha=0.9, zorder=1,
                   label=f'Radio Horizon: {d_los:.0f} km')

        # Shade below audibility threshold (faint red)
        ax.axhspan(-300, 0, alpha=0.06, color='#FFD54F', zorder=0)

        # Labels
        title = f"{r['tx_station']} > {r['target']}"
        subtitle = (f"{r['distance_m']/1000:.0f} km | "
                    f"TX {h_tx:.0f}m | RX {h_rx:.0f}m | "
                    f"{r['freq_mhz']:.1f} MHz | "
                    f"Squint {SQUINT_DEG:.0f}° ({CROSSOVER_dB:.2f} dB) | "
                    f"Globe eq SNR = {r['globe_SNR_eq']:.1f} dB")

        ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight='bold')
        ax.set_xlabel('Distance (km)', fontsize=12)
        ax.set_ylabel('Equisignal SNR (dB)', fontsize=12)
        # For very long paths (>=1000 km), start x-axis at 0 so the
        # legend has room in the upper right without colliding with
        # the intersection points near the middle of the graph.
        x_start = 0 if d_actual >= 1000 else 50
        ax.set_xlim(x_start, d_max)
        ax.set_ylim(-150, 150)
        ax.grid(alpha=0.2, color='gray')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.3)

        fig.tight_layout()

        fname = f"{prefix}_{r['path_id'].lower()}.png"
        outpath = os.path.join(outdir, fname)
        plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        print(f"  Saved: {fname}")


def generate_per_path_graphs_watts(results, outdir=None):
    """
    Generate individual graphs in LINEAR WATTS (not dB).

    These show the same data as the dB graphs but on a linear power
    scale, making the exponential decay in the shadow zone visually
    obvious.  On a linear scale, the signal crashes toward zero
    instead of appearing as a straight line.

    The y-axis shows received power in watts.  The noise floor is
    shown as a horizontal line so you can see how the signal compares
    to noise in absolute terms.
    """
    if outdir is None:
        outdir = os.path.dirname(__file__)

    plt.style.use('dark_background')

    graph_paths = [r for r in results
                   if r["type"] in ("operational", "measurement", "telefunken", "intercept", "trial")]

    for r in graph_paths:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                        height_ratios=[2, 1],
                                        sharex=True)

        d_actual = r["distance_m"] / 1000.0
        h_tx = r["tx_alt_m"]
        h_rx = r["rx_alt_m"]
        freq = r["freq_mhz"] * 1e6
        rx_gain = r["rx_gain_dbi"]
        ground = r.get("ground_type", "land")

        # For long paths (>=1000 km), extend d_max further so the
        # legend in upper right has room away from the intersection
        # points near the middle of the graph.
        if d_actual >= 1000:
            d_max = d_actual + 400
            d_min_plot = 10   # start curves near 0 to fill the axis
        else:
            d_max = max(d_actual + 200, 900)
            d_min_plot = 50
        distances_km = np.arange(d_min_plot, d_max + 1, 5)

        # Compute power curves in WATTS (not dB)
        flat_watts = []
        globe_watts = []

        for dk in distances_km:
            dm = dk * 1000.0
            itu_r = itu_diffraction_loss(dm, h_tx, h_rx, freq, ground=ground)
            f = link_budget(dm, 0.0, freq, rx_gain)
            g = link_budget(dm, itu_r["loss_dB"], freq, rx_gain)
            flat_watts.append(10.0 ** (f["P_rx_dBW"] / 10.0))
            globe_watts.append(10.0 ** (g["P_rx_dBW"] / 10.0))

        flat_watts = np.array(flat_watts)
        globe_watts = np.array(globe_watts)

        # Noise floor in watts
        noise_watts = 10.0 ** (N_FLOOR_dBW / 10.0)

        # Computed values at path distance for legend
        flat_pk_w  = 10.0 ** (r["flat_SNR_peak"] / 10.0) * noise_watts
        globe_pk_w = 10.0 ** (r["globe_SNR_peak"] / 10.0) * noise_watts

        # ---- TOP PANEL: Linear watts (the exponential curves) ----
        ax1.plot(distances_km, flat_watts, color='#4CAF50', linewidth=2.5,
                 label=f'Flat: {flat_pk_w:.2e} W', linestyle='-')
        ax1.plot(distances_km, globe_watts, color='#FF1493', linewidth=2.5,
                 label=f'Globe: {globe_pk_w:.2e} W', linestyle='-')

        ax1.axhline(y=noise_watts, color='#FFD54F', linewidth=2,
                    linestyle='-', label=f'Noise floor: {noise_watts:.2e} W',
                    zorder=5)

        # Mark path distance and LoS on top panel
        d_los = line_of_sight(h_tx, h_rx) / 1000.0
        ax1.axvline(x=d_actual, color='white', linewidth=2, linestyle='-',
                    alpha=0.8, zorder=6)
        ax1.text(d_actual + 5, max(flat_watts) * 0.85,
                 f'{d_actual:.0f} km', fontsize=11, color='white',
                 fontweight='bold', va='top')
        ax1.axvline(x=d_los, color='white', linewidth=1.5, linestyle=':',
                    alpha=0.9, zorder=4)
        ax1.text(d_los + 5, max(flat_watts) * 0.15,
                 f'LoS {d_los:.0f} km', fontsize=10, color='white',
                 fontweight='bold', va='bottom')

        y_max = max(flat_watts) * 1.2
        ax1.set_ylim(-y_max * 0.03, y_max)
        ax1.set_ylabel('Received Power (watts)', fontsize=11)
        ax1.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax1.grid(alpha=0.2, color='gray')
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.3)

        title = f"{r['tx_station']} > {r['target']}"
        subtitle = (f"{r['distance_m']/1000:.0f} km | "
                    f"TX {h_tx:.0f}m | RX {h_rx:.0f}m | "
                    f"{r['freq_mhz']:.1f} MHz | "
                    f"LINEAR WATTS (not dB)")
        ax1.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight='bold')

        # ---- BOTTOM PANEL: Flat/Globe power ratio ----
        # Shows how many times more power the flat model delivers.
        # Within LoS: ratio = 1 (identical).
        # Beyond LoS: ratio explodes exponentially.
        ratio = flat_watts / np.maximum(globe_watts, 1e-300)

        ax2.plot(distances_km, ratio, color='#E91E63', linewidth=2.5)
        ax2.axhline(y=1.0, color='#888888', linewidth=1, linestyle='--',
                    alpha=0.5)

        # Mark LoS and path distance
        ax2.axvline(x=d_los, color='white', linewidth=1.5, linestyle=':',
                    alpha=0.9, zorder=4)
        ax2.axvline(x=d_actual, color='white', linewidth=2, linestyle='-',
                    alpha=0.8, zorder=6)

        # Annotate the ratio at the actual path distance
        ratio_at_d = flat_pk_w / max(globe_pk_w, 1e-300)
        if ratio_at_d > 1e6:
            ratio_label = f'{ratio_at_d:.1e}x'
        elif ratio_at_d > 100:
            ratio_label = f'{ratio_at_d:,.0f}x'
        else:
            ratio_label = f'{ratio_at_d:.1f}x'
        ax2.text(d_actual + 5, ax2.get_ylim()[1] * 0.7 if ax2.get_ylim()[1] > 10 else ratio_at_d * 0.8,
                 f'Flat delivers\n{ratio_label} more\npower at {d_actual:.0f} km',
                 fontsize=9, color='#E91E63', fontweight='bold', va='top')

        ax2.set_xlabel('Distance (km)', fontsize=12)
        ax2.set_ylabel('Flat / Globe\npower ratio', fontsize=11)
        ax2.set_xlim(50, d_max)
        ax2.set_yscale('log')
        ax2.grid(alpha=0.2, color='gray')
        ax2.set_title('How many times more power flat model delivers vs globe',
                      fontsize=10, color='#AAAAAA')

        fig.tight_layout()

        fname = f"watts_{r['path_id'].lower()}.png"
        outpath = os.path.join(outdir, fname)
        plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        print(f"  Saved: {fname}")


def generate_master_comparison(results, outdir=None):
    """
    Generate a master bar chart comparing all paths.
    """
    if outdir is None:
        outdir = os.path.dirname(__file__)

    plt.style.use('dark_background')

    # Filter to operational + measurement + telefunken
    show = [r for r in results
            if r["type"] in ("operational", "measurement", "telefunken")]

    fig, ax = plt.subplots(figsize=(16, 8))

    labels = []
    globe_eq_vals = []
    flat_eq_vals = []

    for r in show:
        name = f"{r['tx_station'][:6]}>{r['target'][:10]}"
        if r["type"] == "telefunken":
            name = f"TF {r['distance_m']/1000:.0f}km"
        labels.append(name)
        globe_eq_vals.append(r["globe_SNR_eq"])
        flat_eq_vals.append(r["flat_SNR_eq"])

    x = np.arange(len(labels))
    w = 0.35

    bars_flat = ax.bar(x - w/2, flat_eq_vals, w, color='#4CAF50',
                       alpha=0.85, label='Flat equisignal SNR')
    bars_globe = ax.bar(x + w/2, globe_eq_vals, w, color='#FF1493',
                        alpha=0.85, label='Globe equisignal SNR (ITU P.526)')

    # Noise floor and thresholds
    ax.axhline(y=0, color='#FFD54F', linewidth=2, linestyle='--',
               label='Noise floor', zorder=5)
    ax.axhline(y=DETECT_dB, color='#FFEB3B', linewidth=1.4,
               linestyle='--', alpha=0.75,
               label=f'Detection floor ({DETECT_dB:.0f} dB above noise)',
               zorder=5)

    # Value labels
    for bar, val in zip(bars_globe, globe_eq_vals):
        y_off = 2 if val >= 0 else -4
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, val + y_off,
                f'{val:.0f}', ha='center', va=va, fontsize=8,
                fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Equisignal SNR (dB)', fontsize=12)
    ax.set_title('Battle of the Beams — All Paths: Flat vs Globe Equisignal SNR\n'
                 'ITU-R P.526-16 (2025) smooth Earth diffraction',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.3)

    fig.tight_layout()
    outpath = os.path.join(outdir, "itu_master_comparison.png")
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close(fig)
    print(f"  Saved: itu_master_comparison.png")


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print("Loading paths from knickebein_paths.csv...")
    results = analyse_all_paths()

    print("\n")
    print_results(results)

    print("\nGenerating per-path graphs (peak + equisignal)...")
    generate_per_path_graphs(results)

    print("\nGenerating per-path graphs (equisignal only)...")
    generate_per_path_graphs_equisignal_only(results)

    print("\nGenerating master comparison bar chart...")
    generate_master_comparison(results)

    # Save text output
    import io
    buf = io.StringIO()
    import sys
    old_stdout = sys.stdout
    sys.stdout = buf
    print_results(results)
    sys.stdout = old_stdout
    outpath = os.path.join(os.path.dirname(__file__), "itu_analysis_output.txt")
    with open(outpath, "w") as f:
        f.write(buf.getvalue())
    print(f"\nSaved text output to: {outpath}")
    print("Done.")
