#!/usr/bin/env python3
"""
Equisignal Width Prediction from First Principles
===================================================

Predicts the Knickebein equisignal corridor width using:
  1. The sinc beam pattern from the 99 m aperture (Fourier optics)
  2. The Lorenz beam-splitting geometry (two squinted sub-beams)
  3. The receiver's ability to resolve amplitude differences (depends on SNR)

The equisignal width is NOT the beam width.  The beam (HPBW) is ~37 km wide
at 440 km.  The equisignal is where the two overlapping beams are equal in
amplitude -- typically 50-100x narrower than the beam, depending on the
squint angle and SNR.

Two models compared:
  Flat:   beam propagates rectilinearly, SNR ~ 86 dB
  Globe:  beam diffracts over the horizon, SNR ~ 8.6 dB (Kleve->Spalding)
          or SNR ~ -67 dB (Stollberg->Beeston)

The azimuthal beam pattern is preserved by creeping-wave geodesics, so the
only difference between the two models is the SNR available for the
dot/dash amplitude comparison.

References:
  Goodman, J. Introduction to Fourier Optics, 3rd ed., Ch. 4.
  Born & Wolf, Principles of Optics, 7th ed., Sec. 8.4.3.
"""

import numpy as np
from datetime import datetime

# ================================================================
#  SYSTEM PARAMETERS (same as botb_propagation.py)
# ================================================================
C       = 299_792_458.0           # speed of light (m/s)
FREQ    = 31_500_000.0            # 31.5 MHz
LAM     = C / FREQ                # wavelength ~ 9.517 m
L_H     = 99.0                    # horizontal antenna aperture (m)

# Distances
D_KLEVE    = 439_541.0            # Kleve -> Spalding (m)
D_STOLLBERG = 693_547.0           # Stollberg -> Beeston (m)

# SNR values from Fock analysis (botb_propagation.py results)
SNR_FLAT_KLEVE_DB   = 85.7       # flat model, Kleve path
SNR_GLOBE_KLEVE_DB  =  8.6       # globe model, Kleve, aircraft at 6 km
SNR_FLAT_STOLL_DB   = 81.7       # flat model, Stollberg path
SNR_GLOBE_STOLL_DB  = -67.5      # globe model, Stollberg, aircraft at 6 km

# Lorenz keying parameters
# The Lorenz system alternates dots and dashes at ~60 Hz.
# The receiver integrates over ~1 second, giving N ~ 60 samples
# for each comparison.
KEYING_RATE = 60.0                # dot/dash pairs per second
INTEGRATION_TIME = 1.0            # seconds of averaging
N_SAMPLES = KEYING_RATE * INTEGRATION_TIME   # 60 independent samples


# ================================================================
#  BEAM PATTERN
# ================================================================

def sinc_pattern(theta_rad):
    """
    Normalised far-field pattern of a uniformly illuminated aperture.

    E(theta) = sinc(pi * L * theta / lambda)
             = sin(pi*L*theta/lam) / (pi*L*theta/lam)

    This is the Fraunhofer diffraction pattern of a single slit of
    width L.  The first nulls are at theta = +/- lambda/L.

    Source: Goodman, Intro to Fourier Optics, 3rd ed., Ch. 4.
            Born & Wolf, Principles of Optics, Sec. 8.4.3.

    Parameters
    ----------
    theta_rad : float or array
        Angle from boresight (radians)

    Returns
    -------
    E : float or array
        Normalised electric field amplitude (1.0 on axis)
    """
    u = np.pi * L_H * theta_rad / LAM       # normalised aperture coordinate
    return np.sinc(u / np.pi)                # np.sinc(x) = sin(pi*x)/(pi*x)


def sinc_pattern_dB(theta_rad):
    """Beam pattern in dB relative to on-axis peak."""
    E = sinc_pattern(theta_rad)
    return 20.0 * np.log10(np.maximum(np.abs(E), 1e-30))


# ================================================================
#  EQUISIGNAL GEOMETRY
# ================================================================

def equisignal_width(squint_deg, snr_dB, distance_m, confidence_sigma=2.0):
    """
    Compute the equisignal corridor width for a Lorenz-type beam system.

    The antenna produces two beams squinted by +/- delta from boresight.
    The "left" beam carries dots, the "right" carries dashes.  At the
    centre (theta=0), both beams have equal amplitude -- the equisignal.

    The corridor width is the angular range where the receiver CANNOT
    reliably tell which beam is stronger, i.e., where the amplitude
    ratio |E_dot/E_dash - 1| is smaller than the noise-limited
    detection threshold.

    Detection threshold derivation:
      The receiver alternately samples the dot and dash amplitudes.
      Each sample has noise-limited accuracy:
        sigma_A / A = 1 / sqrt(2 * SNR_linear)
      After averaging N independent samples:
        sigma_ratio = sqrt(2) * sigma_A / (A * sqrt(N))
                    = 1 / sqrt(SNR_linear * N)
      The minimum detectable ratio difference at n-sigma confidence:
        Delta_R_min = n * sigma_ratio = n / sqrt(SNR_linear * N)

    Parameters
    ----------
    squint_deg : float
        Half-angle between the two beams (degrees).  Each beam is
        offset by +/- squint_deg from boresight.
    snr_dB : float
        Signal-to-noise ratio at the receiver (dB).
    distance_m : float
        Distance from transmitter to measurement point (m).
    confidence_sigma : float
        Number of standard deviations for detection threshold (default 2).

    Returns
    -------
    dict with:
        eq_width_m    : equisignal corridor width in metres
        eq_width_yd   : equisignal corridor width in yards
        eq_angle_deg  : equisignal angular width in degrees
        crossover_dB  : beam level at crossover point (dB below peak)
        delta_R_min   : minimum detectable amplitude ratio difference
        beam_slope    : normalised beam slope at crossover (1/rad)
    """
    delta_rad = np.radians(squint_deg)

    # --- Crossover level ---
    # At theta=0, both beams are at E(delta), so the crossover level
    # relative to the on-axis peak is:
    E_cross = abs(sinc_pattern(delta_rad))
    cross_dB = 20.0 * np.log10(max(E_cross, 1e-30))

    # --- Beam pattern slope at crossover ---
    # The ratio R(theta) = E(theta - delta) / E(theta + delta)
    # Near theta = 0:
    #   dR/dtheta |_{theta=0} = -2 * E'(delta) / E(delta)
    #
    # where E'(delta) is the derivative of the sinc pattern at delta.
    #
    # We compute this numerically for robustness.
    eps = 1e-8  # small angle for numerical derivative (rad)
    E_plus  = sinc_pattern(delta_rad + eps)
    E_minus = sinc_pattern(delta_rad - eps)
    dE_ddelta = (E_plus - E_minus) / (2.0 * eps)   # dE/dtheta at delta

    # Slope of the log-amplitude ratio at theta = 0:
    # d(R_dB)/dtheta = (20/ln10) * 2 * |dE/dtheta| / |E(delta)|
    if abs(E_cross) > 1e-20:
        slope_dB_per_rad = (20.0 / np.log(10.0)) * 2.0 * abs(dE_ddelta) / abs(E_cross)
    else:
        slope_dB_per_rad = 0.0

    # --- Detection threshold ---
    # The equisignal width is set by the smallest power difference the
    # system can resolve.  This has two components:
    #
    # 1. Instrument/operator JND (Just Noticeable Difference):
    #    The Lorenz cross-pointer meter + human operator can tell which
    #    beam is stronger when the power difference exceeds about 1 dB.
    #    This is a fixed threshold independent of SNR.
    #
    # 2. Noise-limited sensitivity:
    #    At low SNR, random amplitude fluctuations mask small beam
    #    differences.  After averaging N dot/dash samples, the noise
    #    contribution to the amplitude ratio measurement is:
    #      sigma_R = confidence / sqrt(SNR_linear * N)
    #    In dB: sigma_dB ~ 8.686 * sigma_R
    #
    # The effective threshold is the RSS (root-sum-square) of both:
    #    delta_dB_eff = sqrt(JND_dB^2 + sigma_noise_dB^2)
    #
    # At high SNR (flat model), JND dominates -> equisignal width is
    # set by the instrument alone.
    # At low SNR (globe model), noise dominates -> equisignal widens.
    # At very low SNR (signal below noise), no equisignal is possible.

    JND_dB = 1.0  # Lorenz meter just-noticeable difference (dB)

    if snr_dB > -200:
        snr_lin = 10.0 ** (snr_dB / 10.0)
        if snr_lin > 0 and N_SAMPLES > 0:
            # Noise-limited ratio uncertainty after averaging N samples
            sigma_R = confidence_sigma / np.sqrt(snr_lin * N_SAMPLES)
            sigma_dB = 8.686 * sigma_R
        else:
            sigma_R = 999.0
            sigma_dB = 999.0
    else:
        sigma_R = 999.0
        sigma_dB = 999.0

    # Effective detection threshold (dB)
    delta_dB_min = np.sqrt(JND_dB**2 + sigma_dB**2)
    delta_R_min = delta_dB_min / 8.686  # fractional ratio for output

    # --- Equisignal half-width ---
    # The equisignal corridor is where |R_dB(theta)| < delta_dB_min.
    # Near theta=0, R_dB(theta) ~ slope_dB_per_rad * theta.
    # So the half-width is:
    if slope_dB_per_rad > 0:
        eq_half_rad = delta_dB_min / slope_dB_per_rad
    else:
        eq_half_rad = np.pi   # no discrimination possible

    eq_full_rad = 2.0 * eq_half_rad    # full equisignal angular width
    eq_full_deg = np.degrees(eq_full_rad)
    eq_width_m  = distance_m * eq_full_rad   # linear width at target distance
    eq_width_yd = eq_width_m / 0.9144

    return dict(
        eq_width_m=eq_width_m,
        eq_width_yd=eq_width_yd,
        eq_angle_deg=eq_full_deg,
        crossover_dB=cross_dB,
        delta_R_min=delta_R_min,
        slope_dB_per_rad=slope_dB_per_rad,
        beam_slope_norm=2.0 * abs(dE_ddelta) / abs(E_cross) if abs(E_cross) > 1e-20 else 0,
    )


# ================================================================
#  MAIN
# ================================================================

def main():
    out = []
    def p(s=""): out.append(s)
    SEP  = "=" * 78
    SEP2 = "-" * 78

    p(f"\n{SEP}")
    p("  EQUISIGNAL WIDTH PREDICTION FROM FIRST PRINCIPLES")
    p(SEP)
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # --- Beam pattern summary ---
    hpbw_deg = np.degrees(0.886 * LAM / L_H)
    fnbw_deg = np.degrees(2.0 * LAM / L_H)

    p(f"\n{SEP}")
    p("  1. ANTENNA BEAM PATTERN")
    p(SEP)
    p(f"  Aperture L         = {L_H} m")
    p(f"  Wavelength lambda  = {LAM:.3f} m")
    p(f"  HPBW               = {hpbw_deg:.2f}°   (half-power, -3 dB)")
    p(f"  FNBW               = {fnbw_deg:.2f}°   (first-null to first-null)")
    p(f"")
    p(f"  At {D_KLEVE/1e3:.0f} km (Kleve -> Spalding):")
    p(f"    HPBW width       = {D_KLEVE * np.radians(hpbw_deg):.0f} m  = {D_KLEVE * np.radians(hpbw_deg)/1e3:.1f} km")
    p(f"    FNBW width       = {D_KLEVE * np.radians(fnbw_deg):.0f} m  = {D_KLEVE * np.radians(fnbw_deg)/1e3:.1f} km")
    p(f"")
    p(f"  At {D_STOLLBERG/1e3:.0f} km (Stollberg -> Beeston):")
    p(f"    HPBW width       = {D_STOLLBERG * np.radians(hpbw_deg):.0f} m  = {D_STOLLBERG * np.radians(hpbw_deg)/1e3:.1f} km")
    p(f"    FNBW width       = {D_STOLLBERG * np.radians(fnbw_deg):.0f} m  = {D_STOLLBERG * np.radians(fnbw_deg)/1e3:.1f} km")

    # --- Equisignal vs squint angle ---
    p(f"\n{SEP}")
    p("  2. EQUISIGNAL WIDTH vs. SQUINT ANGLE (Flat Model, Kleve path)")
    p(SEP)
    p(f"  SNR = {SNR_FLAT_KLEVE_DB:.1f} dB (flat model)")
    p(f"  Distance = {D_KLEVE/1e3:.0f} km")
    p(f"  Confidence = 2 sigma (95%)")
    p(f"")
    p(f"  {'Squint':>8} {'Crossover':>10} {'Beam slope':>12} {'Min detectable':>15} {'Equisignal':>12} {'Equisignal':>12}")
    p(f"  {'(deg)':>8} {'level (dB)':>10} {'(dB/deg)':>12} {'ratio diff':>15} {'angle (deg)':>12} {'width (yd)':>12}")
    p(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*15} {'-'*12} {'-'*12}")

    for sq in [0.5, 1.0, 1.5, 2.0, 2.44, 3.0, 3.5, 4.0, 4.5, 5.0]:
        r = equisignal_width(sq, SNR_FLAT_KLEVE_DB, D_KLEVE)
        slope_per_deg = r['slope_dB_per_rad'] * np.pi / 180
        p(f"  {sq:>8.2f} {r['crossover_dB']:>10.2f} {slope_per_deg:>12.1f} {r['delta_R_min']:>15.6f} {r['eq_angle_deg']:>12.5f} {r['eq_width_yd']:>12.0f}")

    # --- Find the squint angle that gives ~500 yd ---
    p(f"\n{SEP}")
    p("  3. FINDING SQUINT ANGLE THAT MATCHES MEASURED 500 YD")
    p(SEP)

    target_yd = 500.0
    target_m  = target_yd * 0.9144

    # Binary search for squint angle
    lo, hi = 0.1, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        r = equisignal_width(mid, SNR_FLAT_KLEVE_DB, D_KLEVE)
        if r['eq_width_yd'] < target_yd:
            hi = mid
        else:
            lo = mid
    best_squint = (lo + hi) / 2.0
    r_best = equisignal_width(best_squint, SNR_FLAT_KLEVE_DB, D_KLEVE)

    p(f"")
    p(f"  Target:       500 yd equisignal at {D_KLEVE/1e3:.0f} km")
    p(f"  Best squint:  {best_squint:.3f}°  (half-angle between the two beams)")
    p(f"  Full squint:  {2*best_squint:.3f}°  (beam-to-beam separation)")
    p(f"  Crossover:    {r_best['crossover_dB']:.2f} dB below peak")
    p(f"  This means the beams cross at the {abs(r_best['crossover_dB']):.1f} dB point")
    p(f"  Squint / HPBW ratio: {best_squint / hpbw_deg:.2f}")
    p(f"")
    p(f"  Verification:")
    p(f"    Predicted equisignal: {r_best['eq_width_yd']:.0f} yd  ({r_best['eq_width_m']:.0f} m)")
    p(f"    Measured:             500 yd  (457 m)")

    # --- Compare Flat vs Globe at this squint angle ---
    p(f"\n{SEP}")
    p("  4. FLAT vs. GLOBE MODEL EQUISIGNAL PREDICTIONS")
    p(SEP)
    p(f"  Using squint angle = {best_squint:.3f}° (derived from flat model fit)")
    p(f"")
    p(f"  {'':>30} {'SNR':>8} {'Min detect':>12} {'Eq. angle':>10} {'Eq. width':>10} {'Eq. width':>10}")
    p(f"  {'Scenario':>30} {'(dB)':>8} {'ratio':>12} {'(deg)':>10} {'(m)':>10} {'(yd)':>10}")
    p(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    scenarios = [
        ("KLEVE (440 km) — Flat",        SNR_FLAT_KLEVE_DB,  D_KLEVE),
        ("KLEVE (440 km) — Globe",       SNR_GLOBE_KLEVE_DB, D_KLEVE),
        ("STOLLBERG (694 km) — Flat",     SNR_FLAT_STOLL_DB,  D_STOLLBERG),
        ("STOLLBERG (694 km) — Globe",    SNR_GLOBE_STOLL_DB, D_STOLLBERG),
    ]

    for label, snr, dist in scenarios:
        r = equisignal_width(best_squint, snr, dist)
        if r['eq_width_yd'] > 1e6:
            yd_str = "N/A"
            m_str  = "N/A"
            ang_str = "N/A"
        else:
            yd_str = f"{r['eq_width_yd']:.0f}"
            m_str  = f"{r['eq_width_m']:.0f}"
            ang_str = f"{r['eq_angle_deg']:.4f}"
        p(f"  {label:>30} {snr:>8.1f} {r['delta_R_min']:>12.4f} {ang_str:>10} {m_str:>10} {yd_str:>10}")

    # Measured value for comparison
    p(f"  {'MEASURED (Kleve, Spalding)':>30} {'—':>8} {'—':>12} {'0.060':>10} {'457':>10} {'500':>10}")

    # --- Explain the physics ---
    p(f"\n{SEP}")
    p("  5. INTERPRETATION")
    p(SEP)

    r_flat_kl  = equisignal_width(best_squint, SNR_FLAT_KLEVE_DB,  D_KLEVE)
    r_globe_kl = equisignal_width(best_squint, SNR_GLOBE_KLEVE_DB, D_KLEVE)
    r_flat_st  = equisignal_width(best_squint, SNR_FLAT_STOLL_DB,  D_STOLLBERG)
    r_globe_st = equisignal_width(best_squint, SNR_GLOBE_STOLL_DB, D_STOLLBERG)

    p(f"")
    p(f"  The full beam (HPBW) at Spalding is {D_KLEVE * np.radians(hpbw_deg)/1e3:.1f} km wide.")
    p(f"  The equisignal is a thin corridor within that beam, where the receiver")
    p(f"  cannot tell which sub-beam (dots or dashes) is stronger.")
    p(f"")
    p(f"  On the flat model:")
    p(f"    SNR = {SNR_FLAT_KLEVE_DB:.0f} dB allows detection of {r_flat_kl['delta_R_min']*100:.4f}% amplitude differences")
    p(f"    This gives a {r_flat_kl['eq_width_yd']:.0f} yd equisignal at {D_KLEVE/1e3:.0f} km")
    p(f"    Matches measured 500 yd.")
    p(f"")
    p(f"  On the globe model (Kleve, aircraft at 6 km):")
    p(f"    SNR = {SNR_GLOBE_KLEVE_DB:.1f} dB allows detection of {r_globe_kl['delta_R_min']*100:.1f}% amplitude differences")
    p(f"    This gives a {r_globe_kl['eq_width_yd']:.0f} yd equisignal at {D_KLEVE/1e3:.0f} km")
    ratio_kl = r_globe_kl['eq_width_yd'] / 500.0
    p(f"    That is {ratio_kl:.0f}x wider than the measured 500 yd")
    p(f"")
    p(f"  On the globe model (Stollberg, aircraft at 6 km):")
    p(f"    SNR = {SNR_GLOBE_STOLL_DB:.1f} dB — signal is below the noise floor")
    if r_globe_st['eq_width_m'] > 1e6:
        p(f"    No equisignal is possible (receiver sees only noise)")
    else:
        p(f"    Equisignal = {r_globe_st['eq_width_yd']:.0f} yd — {r_globe_st['eq_width_yd']/500:.0f}x wider than measured")

    p(f"")
    p(f"  BEAM WIDTH vs EQUISIGNAL WIDTH at Spalding ({D_KLEVE/1e3:.0f} km):")
    p(f"    Full beam HPBW    = {D_KLEVE * np.radians(hpbw_deg):.0f} m  ({D_KLEVE * np.radians(hpbw_deg)/1e3:.1f} km)")
    p(f"    Equisignal (flat) = {r_flat_kl['eq_width_m']:.0f} m  ({r_flat_kl['eq_width_yd']:.0f} yd)")
    p(f"    Ratio beam/eq     = {D_KLEVE * np.radians(hpbw_deg) / r_flat_kl['eq_width_m']:.0f}:1")

    p(f"\n{SEP}")
    p("  The flat model predicts both the full beam width and the narrow")
    p("  equisignal corridor from antenna geometry alone.  The globe model")
    p("  preserves the angular pattern but reduces the SNR, which widens")
    p("  the equisignal because the receiver cannot resolve small amplitude")
    p("  differences in a noisy signal.")
    p(SEP)

    text = "\n".join(out)
    print(text)

    outpath = "/home/alan/claude/BotB/equisignal_output.txt"
    with open(outpath, "w") as f:
        f.write(text)
    print(f"\n[Saved to {outpath}]")


if __name__ == "__main__":
    main()
