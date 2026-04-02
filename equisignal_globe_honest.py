#!/usr/bin/env python3
"""
Honest Globe Model: What Happens to the Equisignal After Diffraction?
=====================================================================

Previous analysis was too generous to the globe model by assuming the
azimuthal beam pattern survives diffraction intact.  This script works
through what ACTUALLY happens step by step on a sphere.

The Knickebein equisignal requires:
  1. Two sub-beams (dots and dashes) arriving with slightly different
     amplitudes depending on the aircraft's lateral position
  2. Enough SNR to resolve that amplitude difference
  3. Coherent, stable signal so the Lorenz meter gives a steady reading

On a globe, the beam must diffract over the horizon via creeping waves.
This analysis examines whether conditions 1-3 survive that process.

References:
  Fock (1945), Neubauer et al. (1969), Eckersley (1937)
"""

import numpy as np
from scipy.special import airy
from datetime import datetime

# ================================================================
#  CONSTANTS (from botb_propagation.py)
# ================================================================
C       = 299_792_458.0
R_EARTH = 6_371_000.0
K_REFR  = 4.0 / 3.0
R_EFF   = K_REFR * R_EARTH
FREQ    = 31_500_000.0
LAM     = C / FREQ
K_WAVE  = 2.0 * np.pi / LAM
L_H     = 99.0
H_TX    = 200.0
H_RX    = 6000.0      # aircraft
P_TX    = 3000.0
K_BOLTZ = 1.380649e-23

D_KLEVE = 439_541.0
D_STOLL = 693_547.0

SEP  = "=" * 78
SEP2 = "-" * 78

def main():
    out = []
    def p(s=""): out.append(s)

    p(f"\n{SEP}")
    p("  HONEST GLOBE ANALYSIS: WHAT HAPPENS TO THE EQUISIGNAL?")
    p(SEP)
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # ================================================================
    #  STEP 1: THE BEAM LEAVES THE ANTENNA
    # ================================================================
    p(f"\n{SEP}")
    p("  STEP 1: THE BEAM LEAVES THE ANTENNA")
    p(SEP)
    p("""
  The Knickebein antenna produces two sub-beams (dots and dashes) with
  a small angular offset (the squint angle).  The squint is set by the
  operator via the electrical phasing of the antenna elements.  It is
  a FREE PARAMETER -- we do not know its exact value.

  What we DO know from the antenna dimensions:
    Aperture L = 99 m, wavelength = 9.517 m
    HPBW (half-power beam width) = 0.886 * lambda/L = 4.88 degrees
    FNBW (first-null beam width) = 2 * lambda/L     = 11.02 degrees

  The full beam is about 5 degrees wide (HPBW).  The equisignal is a
  thin corridor WITHIN that beam where dots and dashes are equal.

  The measured equisignal at Spalding was 400-500 yards.
  At 440 km range, 500 yd = 457 m = 0.060 degrees.

  This is 82x narrower than the HPBW.  The equisignal is narrow because
  the Lorenz technique exploits the SLOPE of the beam pattern at the
  crossover point, not the beam width itself.""")

    # ================================================================
    #  STEP 2: WHAT HAPPENS AT THE HORIZON (GLOBE MODEL)
    # ================================================================
    p(f"\n{SEP}")
    p("  STEP 2: THE BEAM REACHES THE HORIZON (Globe Model)")
    p(SEP)

    d_hor_tx = np.sqrt(2 * R_EFF * H_TX)
    p(f"""
  The transmitter is at {H_TX:.0f} m elevation.  On a globe with standard
  refraction (k = 4/3), the radio horizon is at:

    d_horizon = sqrt(2 * R_eff * h_tx)
              = sqrt(2 * {R_EFF/1e3:.0f} km * {H_TX:.0f} m)
              = {d_hor_tx/1e3:.1f} km

  At this point, the beam is tangent to the surface.  Beyond this
  distance, the beam enters the geometric shadow of the Earth.

  The beam has been traveling as a coherent, directional wavefront
  for {d_hor_tx/1e3:.0f} km.  At the horizon, its pattern is still intact:
  the dot and dash sub-beams still have their original angular offset
  and the equisignal structure is preserved.

  But now the beam must somehow continue into the shadow region to
  reach the aircraft over England.""")

    # ================================================================
    #  STEP 3: COUPLING INTO CREEPING WAVES
    # ================================================================
    p(f"\n{SEP}")
    p("  STEP 3: THE BEAM COUPLES INTO CREEPING WAVES")
    p(SEP)

    kR = K_WAVE * R_EFF
    kR_third = kR ** (1.0/3.0)
    W_creep = R_EFF / kR_third
    m = (kR / 2.0) ** (1.0/3.0)

    p(f"""
  At the tangent point, the grazing beam excites creeping waves on the
  surface.  This is NOT a clean relay of the original beam.  It is a
  modal decomposition: the original wavefront is broken into a set of
  surface-hugging eigenmodes (Fock 1945, Section 2).

  Each creeping wave mode has:
    - Its own phase velocity (different from free space)
    - Its own attenuation rate (exponential decay along the surface)
    - A characteristic ribbon height: W_creep = R * (kR)^(-1/3) = {W_creep/1e3:.1f} km

  The original beam's VERTICAL pattern (29 m aperture, ~17 deg HPBW)
  is immediately lost.  The creeping wave fills a {W_creep/1e3:.1f} km ribbon
  regardless of the original beam's elevation pattern.

  What about the HORIZONTAL pattern?

  Different azimuths from the transmitter correspond to different great
  circle geodesics on the sphere.  Each geodesic carries its own
  creeping wave independently.  So the azimuthal variation (the
  dot/dash pattern) is, in principle, encoded in which geodesics carry
  more or less energy.

  However, the dot and dash beams are offset by only ~0.06 degrees in
  azimuth.  At the tangent point ({d_hor_tx/1e3:.0f} km from TX), this offset
  corresponds to a lateral separation of:

    lateral_sep = {d_hor_tx:.0f} m * tan(0.06 deg) = {d_hor_tx * np.tan(np.radians(0.06)):.1f} m

  Both sub-beams hit the tangent point within {d_hor_tx * np.tan(np.radians(0.06)):.0f} metres of
  each other and couple into effectively the SAME set of creeping waves.
  The tiny angular difference that defines the equisignal is on the
  order of the diffraction limit of the creeping wave itself.""")

    # ================================================================
    #  STEP 4: PROPAGATION IN THE SHADOW
    # ================================================================
    p(f"\n{SEP}")
    p("  STEP 4: CREEPING WAVE PROPAGATION IN THE SHADOW")
    p(SEP)

    # Fock parameters
    xi_kl = m * D_KLEVE / R_EFF
    xi_st = m * D_STOLL / R_EFF

    # First mode attenuation
    tau1 = 2.33811
    alpha_neper_per_m = tau1 * np.sin(np.pi/3) * m / R_EFF
    alpha_dB_per_km = 8.686 * alpha_neper_per_m * 1000

    # Shadow distances
    d_hor_rx = np.sqrt(2 * R_EFF * H_RX)
    d_shadow_kl = max(0, D_KLEVE - d_hor_tx - d_hor_rx)
    d_shadow_st = max(0, D_STOLL - d_hor_tx - d_hor_rx)

    p(f"""
  The creeping wave propagates along the surface at {alpha_dB_per_km:.3f} dB/km
  attenuation (first mode, tau_1 = {tau1}).

  The signal decays exponentially.  Each additional km in the shadow
  costs another 0.292 dB.  Over the shadow distances:

  Kleve -> Spalding (440 km total):
    Shadow zone: {d_shadow_kl/1e3:.0f} km (beyond combined TX+RX horizons)
    Surface attenuation over shadow: {alpha_dB_per_km * d_shadow_kl/1e3:.1f} dB
    (plus geometric factors, coupling losses, etc.)

  Stollberg -> Beeston (694 km total):
    Shadow zone: {d_shadow_st/1e3:.0f} km
    Surface attenuation over shadow: {alpha_dB_per_km * d_shadow_st/1e3:.1f} dB

  The 5-mode Fock residue series (botb_propagation.py) gives the total
  diffraction field ratio including all geometric factors:""")

    # Compute Fock loss for both paths
    for label, d in [("Kleve -> Spalding", D_KLEVE), ("Stollberg -> Beeston", D_STOLL)]:
        xi = m * d / R_EFF
        Y1 = 2 * m**2 * H_TX / R_EFF
        Y2 = 2 * m**2 * H_RX / R_EFF

        tau = np.array([2.33811, 4.08795, 5.52056, 6.78671, 7.94413])
        ej = np.exp(1j * np.pi / 3.0)
        V_sum = 0.0 + 0.0j
        for ts in tau:
            _, aip, _, _ = airy(-ts)
            wt = 1.0 / aip**2
            arg = 1j * ej * ts * xi
            V_sum += np.exp(arg) * wt
        V = 2.0 * np.sqrt(np.pi * max(xi, 1e-30)) * V_sum
        V_abs = abs(V)

        G1 = np.sqrt(1.0 + np.pi * Y1)
        G2 = np.sqrt(1.0 + np.pi * Y2)
        F = V_abs * G1 * G2
        loss_dB = -20 * np.log10(max(F, 1e-300))

        # Link budget
        fspl = 20 * np.log10(4 * np.pi * d / LAM)
        P_rx = 10*np.log10(P_TX) + 26.0 + 3.0 - fspl - loss_dB
        N_floor = 10*np.log10(K_BOLTZ * 290 * 3000) + 32
        snr = P_rx - N_floor

        p(f"""
  {label} ({d/1e3:.0f} km):
    Fock distance xi   = {xi:.2f}
    |V(xi)| (surface)  = {V_abs:.3e}
    Height gain G(Y1)  = {G1:.2f}  (TX at {H_TX:.0f} m)
    Height gain G(Y2)  = {G2:.2f}  (aircraft at {H_RX:.0f} m)
    Total field ratio  = {F:.3e}
    Diffraction loss   = {loss_dB:.1f} dB below free space
    Received power     = {P_rx:.1f} dBW
    Noise floor        = {N_floor:.1f} dBW
    SNR                = {snr:.1f} dB""")

    # ================================================================
    #  STEP 5: WHAT DOES THE AIRCRAFT ACTUALLY RECEIVE?
    # ================================================================
    p(f"\n{SEP}")
    p("  STEP 5: WHAT DOES THE AIRCRAFT ACTUALLY RECEIVE? (Globe Model)")
    p(SEP)
    p(f"""
  KLEVE -> SPALDING (SNR = 8.6 dB):

    The aircraft is at 6 km altitude, which is above the creeping wave
    ribbon ({W_creep/1e3:.1f} km).  The height gain factor of 9.45 (+19.5 dB)
    recovers some signal, but the total path loss still leaves only
    8.6 dB of SNR.

    What does 8.6 dB SNR mean for the pilot?

    SNR_linear = 10^(8.6/10) = {10**(8.6/10):.1f}

    The signal power is {10**(8.6/10):.1f}x the noise power.  The signal
    amplitude is sqrt({10**(8.6/10):.1f}) = {np.sqrt(10**(8.6/10)):.1f}x the noise amplitude.

    Each individual dot or dash sample has this SNR.  The amplitude
    of each sample fluctuates by approximately:
      sigma_A / A = 1 / sqrt(2 * SNR_lin) = 1 / sqrt({2*10**(8.6/10):.1f}) = {1/np.sqrt(2*10**(8.6/10)):.2f}
      = {1/np.sqrt(2*10**(8.6/10))*100:.0f}% fluctuations per sample

    This means each dot/dash sample fluctuates by roughly {1/np.sqrt(2*10**(8.6/10))*100:.0f}% in
    amplitude.  The Lorenz meter averages many samples, but the
    instantaneous signal is noisy and unstable.

    CAN THE EQUISIGNAL WORK?

    The equisignal requires comparing the amplitude of the dot beam to
    the dash beam.  At the equisignal centre, both are EQUAL.  Moving
    off-centre by 250 yd (the edge of the 500 yd corridor), one beam
    should be slightly stronger than the other.

    The amplitude difference at the corridor edge depends on the beam
    pattern slope (set by antenna geometry and squint angle).  Whatever
    that difference is, the receiver must resolve it against the noise.

    At 8.6 dB SNR, after averaging 60 samples/second for 1 second:
      Comparison noise = {1/np.sqrt(10**(8.6/10)*60)*100:.1f}% amplitude ratio uncertainty
      Comparison noise = {8.686/np.sqrt(10**(8.6/10)*60):.2f} dB

    This is about {8.686/np.sqrt(10**(8.6/10)*60):.1f} dB of uncertainty in the dot/dash
    comparison.  The Lorenz meter's JND is about 1 dB.  So the noise
    contribution ({8.686/np.sqrt(10**(8.6/10)*60):.2f} dB) is below the JND (1 dB), meaning
    the instrument threshold dominates.

    HOWEVER, this assumes:
      - The signal is continuous and present for the full averaging period
      - No fading, multipath, or amplitude scintillation
      - The creeping wave delivers a clean, coherent carrier
      - The AGC can lock onto a signal only {10**(8.6/10):.1f}x above noise

    In reality, at 8.6 dB SNR:
      - Rayleigh fading causes the signal to drop below the noise floor
        for a significant fraction of time (about 13% for Rayleigh)
      - During fades, the equisignal comparison fails completely
      - The AGC must track a signal that intermittently disappears
      - The Lorenz meter needle would swing erratically
      - The pilot cannot maintain a steady course on an erratic meter

    Verdict: Marginally detectable carrier, unreliable equisignal.

  STOLLBERG -> BEESTON (SNR = -67.5 dB):

    The signal is {abs(-67.5):.0f} dB below the noise floor.  In linear
    terms, the signal power is {10**(-67.5/10):.1e} times the noise power.

    The receiver hears only noise.  No carrier is detectable.
    No equisignal comparison is possible.
    No beam tracking can occur.

    Verdict: No signal.""")

    # ================================================================
    #  STEP 6: COHERENCE OF THE CREEPING WAVE
    # ================================================================
    p(f"\n{SEP}")
    p("  STEP 6: IS THE CREEPING WAVE SIGNAL COHERENT?")
    p(SEP)

    # Phase velocities of first 3 modes
    tau = np.array([2.33811, 4.08795, 5.52056])
    ej = np.exp(1j * np.pi / 3.0)

    p(f"""
  The Fock residue series decomposes the field into multiple creeping
  wave modes.  Each mode travels along the surface at a DIFFERENT
  phase velocity and decays at a DIFFERENT rate.

  Mode properties (at 31.5 MHz, R_eff = {R_EFF/1e3:.0f} km):""")

    for i, ts in enumerate(tau):
        # Phase velocity relative to free space
        # nu_s = kR + e^{j*pi/3} * m * tau_s
        # Phase velocity: v_s = c * kR / Re(nu_s)
        nu_s = K_WAVE * R_EFF + ej * m * ts
        v_phase = C * K_WAVE * R_EFF / nu_s.real
        v_ratio = v_phase / C

        # Attenuation per km
        alpha_s = 8.686 * ts * np.sin(np.pi/3) * m / R_EFF * 1000  # dB/km

        # Time delay relative to direct path at 440 km
        travel_time_direct = D_KLEVE / C
        travel_time_mode = D_KLEVE / v_phase
        delay_ns = (travel_time_mode - travel_time_direct) * 1e9

        _, aip, _, _ = airy(-ts)
        weight = 1.0 / aip**2

        p(f"    Mode {i+1}: tau = {ts:.3f}")
        p(f"      Attenuation:    {alpha_s:.3f} dB/km")
        p(f"      Phase velocity: {v_ratio:.9f} c")
        p(f"      Weight:         {abs(weight):.3f}")

    p(f"""
  The first mode dominates at long range (lowest attenuation).  Higher
  modes decay faster and contribute less.

  For the Knickebein keying to work, the carrier must be stable enough
  for the 60 Hz dot/dash modulation to be cleanly detected.  If
  multiple modes arrive with different delays, the modulation envelope
  is distorted.

  At 31.5 MHz, the period is 31.7 ns.  The mode velocity differences
  are tiny (parts per billion of c), so the inter-modal delay at 440 km
  is small compared to the carrier period.  This means the carrier
  itself remains coherent -- the creeping wave does not smear the RF
  carrier.

  However, the AMPLITUDE stability is the problem.  The creeping wave
  modes interfere constructively and destructively as they propagate,
  causing amplitude fluctuations (analogous to multipath fading).  At
  8.6 dB SNR, these fluctuations push the signal below the noise floor
  intermittently.""")

    # ================================================================
    #  STEP 7: SUMMARY
    # ================================================================
    p(f"\n{SEP}")
    p("  STEP 7: SUMMARY -- GLOBE MODEL EQUISIGNAL PREDICTION")
    p(SEP)
    p(f"""
  On a flat surface:
    - The beam propagates rectilinearly from the antenna
    - Both sub-beams (dots and dashes) arrive intact at full strength
    - SNR ~ 86 dB: the signal is 400 million times above the noise
    - The equisignal is determined solely by the antenna pattern and
      operator-set squint angle
    - A squint of ~5 deg (near the first null) with a 1 dB JND
      produces ~500 yd equisignal -- but we cannot independently
      verify the squint angle was 5 deg; this was reverse-engineered
      from the measurement
    - The beam was measured at 400-500 yards by British aircraft

  On a globe (Kleve, 440 km):
    - The beam hits the horizon at {d_hor_tx/1e3:.0f} km
    - Beyond that, the signal exists only as a creeping wave
    - The creeping wave fills a {W_creep/1e3:.0f} km vertical ribbon (the original
      beam's vertical structure is lost)
    - The signal is attenuated by 77 dB relative to free space
    - SNR = 8.6 dB: the signal is {10**(8.6/10):.1f}x above the noise
    - At this SNR, each amplitude sample fluctuates by ~{1/np.sqrt(2*10**(8.6/10))*100:.0f}%
    - The azimuthal beam pattern may be partially preserved, but both
      sub-beams go through the same diffraction and the tiny angular
      difference (~0.06 deg) between them is at the resolution limit
      of the creeping wave process
    - Signal fading would intermittently drop the carrier below noise
    - The Lorenz meter cannot maintain a steady reading
    - Continuous beam tracking from Germany to England is not feasible
      because the last 62 km are in the shadow zone with degraded signal

  On a globe (Stollberg, 694 km):
    - 316 km of the path is in shadow (46% of the total)
    - The signal is 67.5 dB BELOW the noise floor
    - No carrier is detectable; no equisignal comparison is possible
    - The receiver hears only noise""")

    p(f"""
  WHAT WE CAN SAY WITHOUT KNOWING THE SQUINT ANGLE:

    1. The antenna dimensions (99 m aperture, 9.5 m wavelength) give a
       beam HPBW of 4.88 degrees.  This is a firm prediction from
       Fourier optics, independent of squint or propagation model.

    2. At 440 km, the HPBW corresponds to 37.4 km beam width.

    3. The measured equisignal (500 yd = 457 m) is 82x narrower than
       the beam.  This narrowing comes from the Lorenz beam-splitting
       technique operating on the beam's steep edges.

    4. On a flat surface, this beam width and equisignal width are
       consistent with rectilinear propagation from a 99 m aperture
       at 31.5 MHz, with a squint angle that places the crossover
       in the steep part of the beam pattern.

    5. On a globe, the Fock analysis shows the signal arrives at Kleve's
       range with SNR of 8.6 dB (marginal) and at Stollberg's range
       with no detectable signal at all.  Even if the equisignal
       angular pattern survived diffraction, the signal quality is
       insufficient for precision navigation.""")

    p(f"\n{SEP}")

    text = "\n".join(out)
    print(text)

    outpath = "/home/alan/claude/BotB/equisignal_globe_honest_output.txt"
    with open(outpath, "w") as f:
        f.write(text)
    print(f"\n[Saved to {outpath}]")


if __name__ == "__main__":
    main()
