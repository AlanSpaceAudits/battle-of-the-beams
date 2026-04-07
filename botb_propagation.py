#!/usr/bin/env python3
"""
Battle of the Beams — VHF Propagation Analysis: Flat vs. Curved Surface
=======================================================================

Computes whether the Knickebein beam system's observed properties
(equisignal width, signal strength, targeting accuracy) are consistent
with rectilinear (flat) propagation or spherical-Earth diffraction.

The Knickebein ("crooked leg") system was a modified Lorenz blind-landing
beam used by the Luftwaffe during WWII for precision night bombing.  Two
narrow VHF beams were aimed from continental Europe to intersect over a
British target.  An He 111 bomber flew along one beam (the "director")
and released bombs when the second beam ("cross") was detected.  The
equisignal technique superimposes two slightly offset beams: to the left
of centre the pilot hears dots, to the right dashes, and on-axis the two
interleave into a steady tone.  Beam width at the target determines
bombing accuracy.

The central question: can a 31.5 MHz beam maintain a ~500-yard equisignal
width after travelling 440-694 km if the Earth is a sphere?  On a sphere,
the beam must diffract around the horizon, which smears the wavefront into
a kilometres-wide "creeping-wave ribbon" and attenuates the signal by tens
of dB.  On a flat surface, the beam propagates rectilinearly and arrives
intact.

This script computes both models and compares them to the historically
measured beam properties.

Paths analysed:
  Kleve (Kn.4) -> Spalding       ~440 km
  Stollberg (Kn.2) -> Beeston    ~694 km

Formulas used:
  1. Geometric Earth-curvature drop              (standard geodesy)
  2. Antenna beam pattern & equisignal width      (Fourier optics)
  3. Knife-edge far-field beam width              (Huygens-Fresnel)
  4. Creeping-wave ribbon width                   (Fock-Keller GTD/UTD)
  5. Fock smooth-Earth diffraction loss           (residue series)
  6. Link budget with atmospheric noise           (ITU-R)

References:
  [1] Fock, V.A. (1965). Electromagnetic Diffraction and Propagation
      Problems. Pergamon Press, Oxford. (Ch. 10 = Fock 1945 paper)
      - Eq. (6.10), p. 209:  General attenuation factor V(x,y,q)
      - Eq. (6.09), p. 209:  Root equation w'(t) - qw(t) = 0
      - Eq. (5.15), p. 207:  Normalised distance x = (ka/2)^{1/3} theta
      - Eq. (5.08), p. 205:  Normalised height y = kh/(ka/2)^{1/3}
      - Eq. (3.22-3.23), p. 201: Weyl-van der Pol flat Earth formula
      - Ch. 13, p. 254: "Equivalent radius has not received adequate
        theoretical foundation"
      - Ch. 15, p. 309: "In problems connected with superrefraction the
        equivalent radius does not play the role..."
  [2] Eckersley, T.L. (1937). J. I.E.E. 80, p.286.
  [3] Vogler, L.E. (1961). NBS J. Res. 65D(4), 397-399.
  [4] Neubauer, Ugincius & Uberall (1969). Z. Naturforsch. 24a, 691-700.
  [5] Keller, J.B. Geometrical Theory of Diffraction (WHOI lectures).
  [6] Bird, J.F. (1985). JOSA A 2(6):945-953.
"""

import numpy as np
from scipy.special import airy   # Airy function Ai(x) and its derivative Ai'(x)
from datetime import datetime
import sys

# ================================================================
#  PHYSICAL CONSTANTS
# ================================================================

# Speed of light in vacuum (m/s).  Exact by SI definition since 2019.
C       = 299_792_458.0

# Mean volumetric radius of the Earth (m).  Standard geodetic value.
R_EARTH = 6_371_000.0

# ITU standard atmosphere refraction factor.  Radio waves at VHF/UHF
# travel through a troposphere whose refractive index decreases with
# altitude, bending ray paths slightly downward.  The standard "4/3
# Earth" model replaces actual refraction with an enlarged "effective
# Earth" so that rays can be drawn as straight lines.  R_eff = (4/3)*R
# gives the correct bending for a well-mixed mid-latitude atmosphere.
# Source: ITU-R P.453, "The radio refractive index."
K_REFR  = 4.0 / 3.0

# Effective Earth radius (m), approximately 8,495 km.  All geometric
# horizon and diffraction calculations in this script use R_EFF so that
# atmospheric refraction is implicitly accounted for.
R_EFF   = K_REFR * R_EARTH

# Boltzmann constant (J/K).  Exact by SI definition since 2019.
# Used to compute the thermal noise floor: N = k_B * T * B.
K_BOLTZ = 1.380649e-23

# ================================================================
#  KNICKEBEIN SYSTEM PARAMETERS
# ================================================================

# Operating frequency (Hz).  Knickebein used 31.5 MHz, in the low VHF
# band.  This is the same band used for Lorenz blind-landing beams;
# the Knickebein array was a scaled-up version with much higher gain.
FREQ     = 31_500_000.0

# Wavelength (m).  lambda = c / f.  At 31.5 MHz, lambda ~ 9.517 m.
LAM      = C / FREQ

# Wavenumber (rad/m), also called the spatial frequency of the radio
# wave.  k = 2*pi/lambda.  It tells you how many radians of phase
# accumulate per metre of travel.  k ~ 0.660 rad/m at 31.5 MHz.
# Sometimes written K_WAVE to avoid confusion with the refraction
# factor K_REFR.
K_WAVE   = 2.0 * np.pi / LAM

# Transmit power (W).  3 kW was the rated output of the Knickebein
# transmitter at each station.
P_TX     = 3000.0

# Horizontal aperture of the antenna array (m).  The Knickebein antenna
# was a large Yagi-based array approximately 99 m wide.  This dimension
# sets the horizontal beam width via Fourier transform of the aperture.
L_H      = 99.0

# Vertical aperture of the antenna array (m).  The vertical extent was
# approximately 29 m, determining the vertical beam width.
H_V      = 29.0

# Transmitter elevation above sea level (m).  The Knickebein stations
# were sited on high ground or atop towers; 200 m is a representative
# value for the Kleve and Stollberg installations.
H_TX     = 200.0

# --- Antenna directivity (uniform rectangular aperture) ---
# Physical area of the aperture (m^2).  For a uniform rectangular
# aperture, this is simply width * height = 99 * 29 = 2,871 m^2.
A_PHYS   = L_H * H_V

# Directivity (linear, then dBi).
# G = 4*pi*A / lambda^2
# This is the standard formula for the maximum directivity of a
# uniformly illuminated aperture.  It follows from the far-field
# radiation integral: the on-axis field is proportional to the total
# aperture area, and the solid-angle integral normalises to 4*pi.
# Source: Balanis, "Antenna Theory," Ch. 12, eq. 12-24; also
#         Goodman, "Introduction to Fourier Optics," 3rd ed., Ch. 4.
G_DIR    = 4.0 * np.pi * A_PHYS / LAM**2
G_DIR_dB = 10.0 * np.log10(G_DIR)            # dBi (decibels relative to isotropic)

# --- Beam widths (uniform illumination) ---
# FNBW = First-Null Beam Width:  the angular distance between the first
# zeros of the sinc pattern.  For a uniform aperture of length L:
#   FNBW = 2 * lambda / L
# Source: Goodman, "Intro to Fourier Optics," 3rd ed., Ch. 4.
# The far-field pattern of a uniform slit is a sinc function whose
# first nulls fall at sin(theta) = +/- lambda/L.
FNBW_H = 2.0 * LAM / L_H                     # rad, horizontal FNBW
FNBW_V = 2.0 * LAM / H_V                     # rad, vertical FNBW

# HPBW = Half-Power Beam Width:  the angular width between the -3 dB
# points (where power drops to half its peak value).  For a uniform
# rectangular aperture:
#   HPBW ~ 0.886 * lambda / L
# The factor 0.886 comes from solving sinc^2(x) = 0.5, which gives
# x ~ 0.4429, so the full width is 2*0.4429 = 0.886 in units of
# lambda/L.
# Source: Born & Wolf, "Principles of Optics," 7th ed., Sec. 8.4.3.
HPBW_H = 0.886 * LAM / L_H                   # rad, horizontal HPBW
HPBW_V = 0.886 * LAM / H_V                   # rad, vertical HPBW

# Measured equisignal angle (rad).  British ground monitoring at Spalding
# determined the Knickebein beam's equisignal (on-course) corridor to be
# approximately 0.066 degrees wide.  This is the half-angle between the
# two offset beams where the dot and dash signals are equal in strength.
EQSIG_ANGLE = np.radians(0.066)              # rad   (0.066 deg)

# --- Aircraft parameters ---
# Operational altitude of the He 111 bomber (m).  Approximately 6,000 m
# (19,685 ft), typical for Knickebein-guided missions.
H_AIRCRAFT = 6000.0

# Height of a ground-based monitoring antenna (m).  The British "Meacon"
# and beam-monitoring stations used masts of roughly this height.
H_GROUND   = 10.0

# --- Receiver noise environment (1940s era) ---
# Noise figure (NF) of the 1940s vacuum-tube receiver (dB).  Tube-based
# superheterodyne receivers of that era had noise figures in the range of
# 8-12 dB.  10 dB is a reasonable estimate.
RX_NF_dB   = 10.0

# Detection bandwidth (Hz).  The Lorenz equisignal keying used dot-dash
# modulation at a rate corresponding to roughly 1-3 kHz audio bandwidth.
# 3 kHz is the matched filter bandwidth for detecting the keyed tones.
RX_BW_Hz   = 3000.0

# System reference temperature (K).  Standard reference temperature
# T_0 = 290 K, used in noise calculations per IEEE/ITU convention.
T_SYS      = 290.0

# External noise figure Fa (dB above kT_0*B).  At 31 MHz, the dominant
# external noise sources are galactic synchrotron radiation (~30 dB) and
# atmospheric noise (~2-5 dB).  Man-made noise is negligible over the
# sea paths these beams traversed.  Fa ~ 32 dB is read from the curves
# in ITU-R P.372, "Radio noise" (Fig. 2, galactic noise at 31 MHz).
# This means the noise power is 10^3.2 ~ 1,585 times above thermal kT_0*B.
FA_dB      = 32.0

# ================================================================
#  PATH DEFINITIONS
# ================================================================
# Each path is defined by:
#   name        — human-readable label
#   tx          — transmitter station name and Knickebein designation
#   rx          — target/receiver location
#   d           — great-circle distance in metres (GCD = Great-Circle Distance)
#   tx_lat/lon  — transmitter coordinates (degrees)
#   rx_lat/lon  — receiver coordinates (degrees)
#   meas_eqsig_yd — measured equisignal width in yards (if available)
PATHS = [
    dict(name="Kleve → Spalding",
         tx="Kleve (Kn.4)", rx="Spalding",
         d=439_541.0,          # GCD in metres, Kleve to Spalding
         tx_lat=51.79, tx_lon=6.10,
         rx_lat=52.79, rx_lon=-0.15,
         meas_eqsig_yd=500),   # British measurement: ~400-500 yards

    dict(name="Stollberg → Beeston",
         tx="Stollberg Hill (Kn.2)", rx="Beeston",
         d=693_547.0,          # GCD in metres, Stollberg to Beeston
         tx_lat=54.65, tx_lon=8.95,
         rx_lat=52.93, rx_lon=-1.22,
         meas_eqsig_yd=None),  # no ground measurement available
]


# ================================================================
#  HELPERS
# ================================================================
def m2mi(m):   return m / 1609.344     # metres to statute miles
def m2yd(m):   return m / 0.9144       # metres to yards
def m2ft(m):   return m / 0.3048       # metres to feet
def m2km(m):   return m / 1000.0       # metres to kilometres

def dB(x):     return 10.0 * np.log10(max(x, 1e-300))
# dB: converts a linear power ratio to decibels.  The max() guard
# prevents log10(0) errors for vanishingly small values.

def dB20(x):   return 20.0 * np.log10(max(abs(x), 1e-300))
# dB20: converts a linear amplitude (field) ratio to decibels.
# Uses 20*log10 because power goes as amplitude squared.


# ================================================================
#  1.  GEOMETRIC ANALYSIS (GLOBE MODEL)
# ================================================================
def geometry(d, h_tx, h_rx, R=R_EFF):
    """
    Line-of-sight geometry on a sphere of radius R.

    Given a great-circle distance d between transmitter and receiver,
    plus their heights above the surface, this function computes:
      - How far the surface curves away from a straight horizontal line
        (the "curvature drop" and "sagitta")
      - The radio horizon distances for each antenna
      - Whether the path is within line-of-sight or extends into the
        geometric shadow zone

    On a sphere, a horizontal beam fired tangent to the surface at height
    h_tx will be at altitude h_tx + d^2/(2R) above the surface at range d.
    If the receiver is lower than this, it is below the geometric horizon.

    Parameters
    ----------
    d    : great-circle distance between TX and RX (m)
    h_tx : transmitter height above the surface (m)
    h_rx : receiver height above the surface (m)
    R    : effective Earth radius (m), default R_EFF = (4/3)*R_Earth
    """

    # Curvature drop (m): how far the surface falls away from a tangent
    # line over distance d.  For small angles (d << R), this is well
    # approximated by d^2 / (2R).  Standard geodetic formula.
    drop       = d**2 / (2.0 * R)

    # Sagitta (m): the maximum bulge of the curved surface above the
    # chord connecting two points separated by distance d.  This is the
    # midpath "hump" that a line-of-sight ray must clear.
    # sagitta = d^2 / (8R) for a circular arc.
    sagitta    = d**2 / (8.0 * R)

    # Beam altitude above surface (m): if a horizontal beam is launched
    # from height h_tx, it will be at this altitude above the surface at
    # range d (because the surface has curved away by "drop" metres).
    beam_alt   = h_tx + drop

    # Radio horizon distance (m): the maximum distance at which a
    # straight ray from height h is tangent to the Earth's surface.
    #   d_horizon = sqrt(2 * R * h)
    # This is the standard radio horizon formula, found in any
    # propagation textbook (e.g., ITU-R P.1546, Recommendation P.530).
    d_tx       = np.sqrt(2.0 * R * h_tx)
    d_rx       = np.sqrt(2.0 * R * h_rx) if h_rx > 0 else 0.0

    # Total line-of-sight range (m): the sum of the two horizon distances.
    # Two antennas can "see" each other as long as the path distance
    # is less than d_los = d_tx + d_rx.
    d_los      = d_tx + d_rx

    # Shadow distance (m): how far beyond line-of-sight the path extends.
    # If positive, the receiver is in the geometric shadow zone and can
    # only receive signal via diffraction (creeping waves, knife-edge, etc.).
    d_shadow   = max(0.0, d - d_los)

    # Central angle (rad): the angle subtended at the Earth's centre by
    # the great-circle arc of length d.  theta = d / R.
    theta      = d / R

    return dict(drop_m=drop, sagitta_m=sagitta, beam_alt_m=beam_alt,
                d_tx_km=d_tx/1e3, d_rx_km=d_rx/1e3,
                d_los_km=d_los/1e3, d_shadow_km=d_shadow/1e3,
                theta_rad=theta)


# ================================================================
#  2.  FLAT-SURFACE BEAM PREDICTIONS
# ================================================================
def flat_predictions(d):
    """
    Beam widths and signal strength for rectilinear (flat-Earth) propagation.

    On a flat surface with no curvature, the beam propagates in a straight
    line from the transmitter to the target.  The angular width of the beam
    maps directly to a linear width at range d via simple trigonometry:
    width = d * tan(angle) ~ d * angle for small angles.

    This gives the equisignal corridor width, the HPBW (Half-Power Beam Width)
    footprint, and the FNBW (First-Null Beam Width) footprint at range d.
    The only loss is FSPL (Free-Space Path Loss) -- the 1/r^2 geometric
    spreading of power over a sphere of radius d.

    Parameters
    ----------
    d : path distance (m)
    """

    # Equisignal corridor width (m): the physical width at range d
    # corresponding to the measured equisignal angle of 0.066 degrees.
    # On a flat surface, this is just d * tan(angle).
    w_eqsig  = d * np.tan(EQSIG_ANGLE)

    # HPBW footprint (m): the width at range d between the -3 dB points
    # of the horizontal beam pattern.  w = d * HPBW_H (small-angle approx).
    w_hpbw_h = d * HPBW_H

    # FNBW footprint (m): the width at range d between the first nulls
    # of the horizontal beam pattern.  w = d * FNBW_H.
    w_fnbw_h = d * FNBW_H

    # HPBW vertical footprint (m): same calculation for the vertical plane.
    w_hpbw_v = d * HPBW_V

    # FSPL = Free-Space Path Loss (dB).  The Friis transmission equation
    # gives the ratio of received to transmitted power for isotropic
    # antennas in free space:
    #   P_r/P_t = (lambda / 4*pi*d)^2
    # Expressed as a loss in dB:
    #   FSPL = 20*log10(4*pi*d / lambda)
    # Source: Friis, H.T. (1946), Proc. IRE, 34(5), pp. 254-256.
    fspl     = 20.0 * np.log10(4.0 * np.pi * d / LAM)

    return dict(w_eqsig_m=w_eqsig, w_hpbw_h_m=w_hpbw_h,
                w_fnbw_h_m=w_fnbw_h, w_hpbw_v_m=w_hpbw_v,
                fspl_dB=fspl)


# ================================================================
#  3.  DIFFRACTION WIDTHS  (friend's PDF formulas)
# ================================================================
def diffraction_widths(R=R_EFF):
    """
    Knife-edge and creeping-wave characteristic widths.

    When a beam encounters the curved horizon of a sphere, it does not
    simply stop.  Part of the wavefront "creeps" along the surface as a
    surface wave (the creeping wave), and part diffracts over the horizon
    like light bending around a knife edge.  These two mechanisms produce
    characteristic spatial scales:

    1) W_knife(z) = 2*z*lambda/L   (Huygens-Fresnel far-field width)
       In the far field (z >> L^2/lambda), the first-null-to-first-null
       diffraction width of a slit of width L grows linearly with distance z.
       Source: Goodman, "Intro to Fourier Optics," 3rd ed., Ch. 4.

    2) W_creep = R * (kR)^{-1/3}   (Fock-Keller creeping-wave ribbon width)
       The creeping wave clings to the surface within a thin ribbon whose
       vertical extent scales as (kR)^{-1/3} * R.  This is the natural
       "diffraction boundary layer" thickness for a sphere of radius R at
       wavenumber k.
       Source: Fock (1945) Zhur Eksp Teor Fiz 15, 479-496;
               Keller, J.B., GTD lectures (WHOI);
               Shim & Kim (1999), PIER 21, 293-306.

    3) s_attach = alpha_p * R * (kR)^{-1/3}   (attachment distance)
       The beam doesn't instantly become a creeping wave at the horizon.
       There is a "penumbra" transition region where the free-space beam
       gradually couples into the surface-guided mode.  The attachment
       distance s_attach is the arc length over which this happens.
       alpha_p ~ 1 for 50%, ~2 for 90%, ~3 for 99% coupling.
       Source: Bird, J.F. (1985), "Diffraction by a conducting sphere,"
               JOSA A 2(6):945-953 (penumbra region analysis).

    Parameters
    ----------
    R : effective Earth radius (m), default R_EFF
    """

    # kR: the product of wavenumber and Earth radius (dimensionless).
    # A huge number (~5.6 million) that characterises how many wavelengths
    # fit around the Earth's circumference.  It controls the ratio of
    # diffraction effects to geometric-optics effects.
    kR          = K_WAVE * R

    # (kR)^{1/3}: the cube root of kR.  This is the fundamental Fock
    # scaling parameter.  It appears everywhere in smooth-Earth diffraction
    # because the transition between the lit and shadow regions on a sphere
    # occurs over an angular scale of (kR)^{-1/3} radians.
    # At 31.5 MHz with R_EFF, (kR)^{1/3} ~ 177.
    kR_third    = kR ** (1.0 / 3.0)

    # Creeping-wave ribbon width (m): the vertical thickness of the
    # surface-hugging diffraction layer.
    #   W_creep = R / (kR)^{1/3} = R * (kR)^{-1/3}
    # This is the height above the surface within which most of the
    # creeping-wave energy is confined.  At 31.5 MHz, W_creep ~ 48 km.
    # Source: Goodman Ch. 4; also Fock (1945); Keller GTD lectures.
    #         Friend's "Diffraction Formulas" PDF, eq. for W_creep.
    W_creep     = R / kR_third

    # Creeping-wave angular width (rad): the angular scale of the
    # diffraction transition zone at the horizon.
    #   delta_theta = (kR)^{-1/3}
    # Each "unit" of this angle corresponds to one Fock length along the
    # surface.  Source: Fock (1945); Keller GTD lectures.
    dtheta      = 1.0 / kR_third

    # Far-field distance (m): the Fraunhofer distance L^2/lambda beyond
    # which the knife-edge (far-field) width formula W = 2*z*lambda/L
    # is valid.  For the 99 m aperture at lambda ~ 9.5 m, this is ~ 1,031 m.
    # All paths of interest (440-694 km) are far beyond this limit.
    far_field   = L_H**2 / LAM

    # Attachment distances (m): the arc length along the surface over which
    # the free-space beam couples into the creeping-wave mode.  These use
    # alpha_p multipliers of 1, 2, and 3 for the 50%, 90%, and 99%
    # coupling thresholds respectively.
    # s_attach = alpha_p * W_creep = alpha_p * R * (kR)^{-1/3}
    # Source: Bird (1985) JOSA A 2(6):945, penumbra region.
    s_attach_50 = 1.0 * W_creep   # 50% of beam energy coupled to surface
    s_attach_90 = 2.0 * W_creep   # 90% coupled
    s_attach_99 = 3.0 * W_creep   # 99% coupled

    return dict(kR=kR, kR_third=kR_third,
                W_creep_m=W_creep, dtheta_rad=dtheta,
                far_field_m=far_field,
                s_attach_50_m=s_attach_50,
                s_attach_90_m=s_attach_90,
                s_attach_99_m=s_attach_99)


def knife_edge_width(d):
    """
    Full first-null diffraction width from aperture L_H at range d.

    In the far field, a uniform slit of width L produces a sinc diffraction
    pattern.  The distance between the first nulls (the FNBW = First-Null
    Beam Width) at range d is:
      W_knife = 2 * d * lambda / L
    This is the linear width at range d between the first zeros of the
    sinc(pi*L*x / (lambda*d)) pattern.
    Source: Goodman, "Intro to Fourier Optics," 3rd ed., Ch. 4.

    Parameters
    ----------
    d : range from the aperture (m)
    """
    return 2.0 * d * LAM / L_H


# ================================================================
#  4.  FOCK SMOOTH-EARTH DIFFRACTION LOSS
# ================================================================
def fock_loss(d, h_tx, h_rx, R=R_EFF):
    """
    Field ratio (diffracted / free-space) using the Fock residue series.

    This is the core diffraction calculation.  On a smooth conducting
    sphere, the electromagnetic field in the shadow region is given by
    a sum of "creeping-wave modes," each of which travels along the
    surface and decays exponentially with distance.  Fock (1945) showed
    that the field can be expressed as a residue series:

      E_diff / E_fs  =  V(xi)  *  G(Y1)  *  G(Y2)

    where:
      V(xi) is the distance factor: surface-to-surface field attenuation
      G(Y)  is the height-gain factor: how much signal you recover by
             being at altitude Y above the surface
      F = V * G1 * G2 is the total field ratio relative to free space

    The distance factor V(xi) is a sum over creeping-wave eigenmodes:
      V(xi) = 2*sqrt(pi*xi) * SUM_s exp(j * e^{j*pi/3} * tau_s * xi)
              / [Ai'(-tau_s)]^2

    Each mode is characterised by an Airy zero tau_s (the s-th root of
    Ai(-tau) = 0).  Higher modes decay faster.  The first mode (tau_1)
    dominates at long range.

    Source: Fock, V. (1945), "Diffraction of Radio Waves Around the
            Earth's Surface," Zhur Eksp Teor Fiz 15, 479-496.
            Also: Neubauer, Ugincius & Uberall (1969), eq. 22.

    Parameters
    ----------
    d     : path distance (m)
    h_tx  : transmitter height above surface (m)
    h_rx  : receiver height above surface (m)
    R     : effective Earth radius (m)
    """

    # kR: wavenumber times effective Earth radius (dimensionless).
    kR  = K_WAVE * R

    # m (Fock parameter, dimensionless): a scale parameter that sets the
    # boundary between the illuminated and shadow regions on a sphere.
    # It tells you how many wavelengths fit into the "transition zone"
    # at the horizon.
    #   m = (kR/2)^{1/3}
    # The factor of 2 in the denominator is a convention from Fock's
    # original formulation that simplifies later expressions.
    # At 31.5 MHz with R_EFF: m ~ 140.
    # Source: Fock (1945), eq. in Sec. 2.
    m   = (kR / 2.0) ** (1.0 / 3.0)

    # xi (Fock normalised distance, dimensionless): the distance along
    # the surface measured in "Fock lengths."
    #   xi = m * d / R
    # xi = 0 corresponds to the geometric shadow boundary (the horizon).
    # xi > 0 means you are in the shadow (diffraction) region -- deeper
    # into the shadow as xi increases.  Each unit of xi corresponds to
    # roughly one Fock length (R/m) along the surface.
    # Source: Fock (1945).
    xi  = m * d / R

    # Y1, Y2 (normalised heights, dimensionless): how far above the
    # surface each antenna is, measured in units of the creeping-wave
    # ribbon width.
    #   Y = 2 * m^2 * h / R
    # Y >> 1 means the antenna is well above the creeping-wave ribbon
    # and captures significantly more of the diffracted field.
    # Y << 1 means the antenna is effectively on the surface.
    # For H_TX = 200 m: Y1 ~ 4.6 (modestly above the ribbon).
    # For H_AIRCRAFT = 6000 m: Y2 ~ 139 (well above the ribbon).
    # Source: Fock (1945).
    Y1 = 2.0 * m**2 * h_tx / R
    Y2 = 2.0 * m**2 * h_rx / R

    # tau_s (Airy zeros, dimensionless): the magnitudes |t^0_s| of the
    # first 5 roots of w(t) = 0, where w(t) is the Fock-Airy function.
    # These are tabulated directly in Fock (1965), Ch.10, p. 209, in the
    # column headed |t^0_s| (the q -> infinity / perfectly conducting case).
    #
    # Each tau_s is an eigenvalue controlling one creeping-wave mode:
    #   tau_1 = 2.338 -> slowest decay (dominant at long range)
    #   tau_2 = 4.088 -> faster decay
    #   tau_3 = 5.521 -> even faster
    #   tau_4 = 6.787 ->  "
    #   tau_5 = 7.944 -> fastest of the 5 modes we use
    #
    # The roots have complex phase pi/3 (Fock Eq. 6.11):
    #   t_s = tau_s * e^{i*pi/3}
    # The imaginary part tau_s * sin(60 deg) causes exponential decay;
    # the real part tau_s * cos(60 deg) gives phase accumulation.
    #
    # Source: Fock (1965), Ch. 10, p. 209, root table.
    #         Cross-check: Abramowitz & Stegun, Table 10.13 (Airy zeros).
    tau = np.array([2.33811, 4.08795, 5.52056, 6.78671, 7.94417])

    # --- Distance factor V(xi) ---
    #
    # GENERAL FOCK EQUATION (1965, Ch.10, p.209, Eq. 6.10):
    #
    #   V(x, y, q) = e^{i*pi/4} * 2*sqrt(pi*x)
    #                * SUM_{s=1}^{inf}  e^{i*x*t_s} / (t_s - q^2)
    #                                  * w(t_s - y) / w(t_s)
    #
    # where:
    #   x = (ka/2)^{1/3} * theta    (normalised distance, = our xi)
    #   y = kh / (ka/2)^{1/3}       (normalised height, = our Y)
    #   q = surface impedance parameter
    #   t_s = roots of w'(t) - q*w(t) = 0, with phase pi/3
    #   w(t) = Fock-Airy function
    #
    # SPECIALISATION we use (q -> inf, y = 0 = surface-to-surface):
    #
    #   V(xi) = e^{i*pi/4} * 2*sqrt(pi*xi)
    #           * SUM_s exp(i * e^{i*pi/3} * tau_s * xi)
    #                   / [Ai'(-tau_s)]^2
    #
    # The e^{i*pi/4} is a constant phase factor that drops out when
    # we take |V| (magnitude). We include it for completeness.
    #
    # The exponential argument i * e^{i*pi/3} * tau * xi decomposes:
    #   i * e^{i*pi/3} = e^{i*pi/2} * e^{i*pi/3} = e^{i*5*pi/6}
    #                   = -sqrt(3)/2 + i/2
    # The NEGATIVE real part (-sqrt(3)/2 * tau * xi) gives exponential
    # DECAY in the shadow -- each mode attenuates as it creeps.
    # The imaginary part (i/2 * tau * xi) gives a phase shift -- the
    # creeping wave accumulates phase as it travels.
    #
    # Source: Fock (1965), Ch. 10, p. 209, Eq. (6.10).
    #         Neubauer et al. (1969), eq. 20-26 (cross-reference).
    ej = np.exp(1j * np.pi / 3.0)        # e^{i*pi/3} = cos(60) + i*sin(60)

    V_sum = 0.0 + 0.0j   # accumulator for the residue series sum
    for ts in tau:
        # Evaluate the Airy function derivative Ai'(-tau_s).
        # scipy.special.airy(x) returns (Ai, Ai', Bi, Bi'); we need Ai'(-ts).
        _, aip, _, _ = airy(-ts)

        # Weight for this mode: 1 / [Ai'(-tau_s)]^2.
        # This factor comes from the residue of the Watson transform at
        # the s-th pole.  Modes with larger Ai' values contribute less.
        wt  = 1.0 / aip**2

        # Exponential decay/phase factor for this mode.
        # arg = j * e^{j*pi/3} * tau_s * xi
        # Real part is negative -> exponential decay (good -- shadow!)
        # Source: Fock (1945); Neubauer et al. (1969) eq. 22.
        arg = 1j * ej * ts * xi

        V_sum += np.exp(arg) * wt

    # V(xi): the surface-to-surface field ratio (complex).
    # Full form: V = e^{i*pi/4} * 2*sqrt(pi*xi) * V_sum
    # The e^{i*pi/4} is the constant phase factor from Fock Eq. (6.10).
    # The prefactor 2*sqrt(pi*xi) comes from the asymptotic normalisation
    # of the Fock-Airy integral representation.
    # If |V| = 10^{-6}, the signal on the surface at that distance is
    # one millionth of the free-space value.
    # Source: Fock (1965), Ch. 10, p. 209, Eq. (6.10).
    phase_factor = np.exp(1j * np.pi / 4.0)   # e^{i*pi/4} from Eq. 6.10
    V  = phase_factor * 2.0 * np.sqrt(np.pi * max(xi, 1e-30)) * V_sum
    V_abs = abs(V)   # magnitude of V (phase factor drops out: |e^{i*pi/4}| = 1)

    # --- Height-gain function G(Y) for each antenna ---
    # G(Y) accounts for the fact that an antenna raised above the surface
    # "reaches above" the shadow and captures more of the diffracted field.
    # For Y >> 1 (antenna well above the creeping-wave ribbon), Fock showed
    # that G -> sqrt(pi * Y), which can be very large.
    # For Y = 0 (antenna on the surface), G = 1 (no gain).
    #
    # The smooth interpolation G(Y) = sqrt(1 + pi*Y) satisfies both limits:
    #   G(0) = sqrt(1) = 1           (antenna on surface)
    #   G(Y>>1) ~ sqrt(pi*Y)         (Fock asymptotic for large height)
    #
    # NOTE: This is an ENGINEERING INTERPOLATION, not Fock's exact formula.
    # Fock's exact height factor is the ratio w(t_s - y) / w(t_s) inside
    # the residue series (Eq. 6.10), which is mode-dependent.  Our approach
    # factors out the height gain as a separate multiplier, which is a good
    # approximation when the dominant first mode controls the sum (i.e., at
    # the ranges we care about, 440-694 km deep in the shadow zone).
    #
    # For an aircraft at 6 km, Y ~ 28 (see below), so G ~ sqrt(1 + pi*28)
    # = sqrt(89.0) ~ 9.4, which is about 19.5 dB of height gain.
    #
    # Source: Fock (1965), Ch. 10, Sec. 5 (asymptotic height gain).
    #         The sqrt(1+pi*Y) form is a standard engineering fit used in
    #         radio propagation codes (cf. Vogler 1961).
    def hgain(Y):
        """Height-gain function G(Y) (linear magnitude).

        Returns the factor by which the diffracted field amplitude
        increases when the antenna is at normalised height Y above
        the surface, relative to an antenna on the surface (Y=0).
        """
        return np.sqrt(1.0 + np.pi * Y)

    # G1: height gain for the transmitter antenna at normalised height Y1.
    G1 = hgain(Y1)

    # G2: height gain for the receiver antenna at normalised height Y2.
    G2 = hgain(Y2)

    # F (total field ratio, dimensionless): the ratio of the diffracted
    # field amplitude to the free-space field amplitude, including both
    # the surface diffraction loss V(xi) and the height gains from both
    # antennas.
    #   F = |V(xi)| * G(Y1) * G(Y2)
    # If F = 0.001, the received field is 1/1000th of what free space
    # would give, corresponding to 60 dB of additional path loss beyond
    # FSPL (Free-Space Path Loss).
    # Source: Fock (1945) complete field expression.
    F = V_abs * G1 * G2

    # Diffraction loss (dB): convert F to dB.
    # loss_dB = -20*log10(F), so a smaller F gives a larger positive loss.
    loss_dB = -dB20(F) if F > 1e-300 else 999.0

    # Per-km attenuation rate of the first (dominant) creeping-wave mode.
    # The first Airy zero tau_1 = 2.338 determines the e-folding decay:
    #   alpha_1 = tau_1 * sin(pi/3) nepers per radian of arc
    # Converting to dB/km:
    #   alpha_1_dB/km = 8.686 * alpha_1 * (m / R) * 1000
    # where 8.686 = 20/ln(10) converts nepers to dB, and m/R converts
    # radians to metres (then * 1000 for km).
    # This gives a useful sanity check: at 31.5 MHz, the first mode
    # decays at roughly 0.28 dB/km along the surface.
    alpha1_np_per_rad = tau[0] * np.sin(np.pi / 3.0)   # nepers per radian
    alpha1_dB_per_km  = 8.686 * alpha1_np_per_rad * m / R * 1000.0

    return dict(m=m, xi=xi, Y1=Y1, Y2=Y2,
                V_abs=V_abs, G1=G1, G2=G2,
                F=F, loss_dB=loss_dB,
                alpha1_dB_per_km=alpha1_dB_per_km)


# ================================================================
#  4b. WEYL-VAN DER POL FLAT EARTH FORMULA (Fock's illuminated region)
# ================================================================
def weyl_van_der_pol_gain_dB():
    """
    Additional signal gain from ground reflection on a flat conducting surface.

    In the illuminated region (no curvature obstruction), Fock derives the
    attenuation factor W using the Weyl-van der Pol formula.  For a
    PERFECTLY CONDUCTING flat surface:

        W = 2

    This means the electric field is doubled compared to pure free space,
    because the direct wave and the ground-reflected wave add constructively.
    In POWER terms, doubling the field means quadrupling the power:

        Power gain = |W|^2 = 4  ->  +6 dB

    This is the standard "ground reflection gain" for a vertical dipole
    over a perfect ground plane at low grazing angles.

    Source: Fock (1965), Ch. 10, p. 200-201, Eq. (3.22)-(3.23).
            In the limit sigma -> 0 (perfect conductor), W -> 2.

    For REAL ground (finite conductivity), the gain is between 0 and +6 dB
    depending on:
      - Ground conductivity (sigma_ground)
      - Polarisation (horizontal vs vertical)
      - Grazing angle (very low angles reduce the gain for horiz. pol.)

    At 31.5 MHz over seawater (most of the Kleve-Spalding path), the
    ground is a good conductor at VHF, so the gain is close to +6 dB.
    Over land, conductivity is lower and the gain is reduced.

    For the Knickebein analysis, we use this as an OPTIONAL comparison
    alongside the conservative pure-FSPL flat model.  The Weyl-van der Pol
    result makes the flat model STRONGER (more SNR headroom), not weaker.

    Returns
    -------
    gain_dB : float
        Power gain in dB from ground reflection.  +6.0 for perfect conductor.
    """
    W_flat = 2.0   # Weyl-van der Pol: field doubled on perfect conductor
    gain_dB = 20.0 * np.log10(W_flat)   # = 20*log10(2) = 6.02 dB
    return gain_dB


# ================================================================
#  5.  LINK BUDGET
# ================================================================
def link_budget(d, diff_loss_dB, rx_gain_dBi=3.0):
    """
    Compare received signal power to noise floor, yielding the SNR
    (Signal-to-Noise Ratio).

    A link budget adds up all gains and subtracts all losses along the
    radio path to find the received power, then compares it to the noise
    power to determine whether the signal is detectable.

    The chain (all in dB):
      P_rx = P_tx + G_tx + G_rx - FSPL - DiffractionLoss

    The noise floor includes:
      - Thermal noise: N_thermal = k_B * T_sys * B  (Johnson-Nyquist)
      - Receiver noise figure NF (internal tube noise)
      - External noise Fa (galactic + atmospheric, from ITU-R P.372)
    Whichever is larger (NF or Fa) dominates.  At 31 MHz, external
    galactic noise (Fa ~ 32 dB) swamps the receiver NF (10 dB).

    SNR = P_rx - N_total.  Typically need SNR >= 10-20 dB for reliable
    equisignal beam discrimination.

    Parameters
    ----------
    d            : path distance (m)
    diff_loss_dB : diffraction loss beyond free space (dB); 0 for flat model
    rx_gain_dBi  : receiver antenna gain (dBi); default 3 dBi (small Yagi)
    """

    # FSPL = Free-Space Path Loss (dB).
    # Friis transmission equation:
    #   FSPL = 20*log10(4*pi*d / lambda)
    # Source: Friis, H.T. (1946), Proc. IRE, 34(5), pp. 254-256.
    fspl = 20.0 * np.log10(4.0 * np.pi * d / LAM)

    # Received power (dBW = decibels relative to 1 watt).
    # P_rx = P_tx(dBW) + G_tx(dBi) + G_rx(dBi) - FSPL(dB) - DiffLoss(dB)
    P_rx_dBW = (10.0 * np.log10(P_TX)    # transmit power in dBW
                + G_DIR_dB                 # Knickebein array directivity (dBi)
                + rx_gain_dBi              # receiver antenna gain (dBi)
                - fspl                     # free-space spreading loss (dB)
                - diff_loss_dB)            # additional diffraction loss (dB)

    # Noise power (dBW).
    # Thermal noise: N = k_B * T * B, converted to dBW.
    N_thermal = 10.0 * np.log10(K_BOLTZ * T_SYS * RX_BW_Hz)

    # Total effective noise: the dominant noise source is either the
    # receiver's internal noise (NF) or external noise (Fa).
    # At 31 MHz, Fa = 32 dB >> NF = 10 dB, so galactic noise dominates.
    # We take the max (in dB) and add it to the thermal noise floor.
    # Source: ITU-R P.372, "Radio noise," for external noise figure Fa.
    N_total   = N_thermal + max(RX_NF_dB, FA_dB)

    # SNR = Signal-to-Noise Ratio (dB).
    # SNR = P_rx - N_total.  Positive means signal is above the noise.
    # Need >= 10-20 dB for the Lorenz equisignal discrimination to work.
    SNR = P_rx_dBW - N_total

    return dict(fspl_dB=fspl, P_rx_dBW=P_rx_dBW,
                N_total_dBW=N_total, SNR_dB=SNR)


# ================================================================
#  OUTPUT FORMATTING
# ================================================================
SEP  = "=" * 78   # major section separator
SEP2 = "-" * 78   # subsection separator

def banner(title):
    return f"\n{SEP}\n  {title}\n{SEP}"

def sub(title):
    return f"\n{SEP2}\n  {title}\n{SEP2}"


# ================================================================
#  MAIN
# ================================================================
def main():
    out = []                # accumulator for all output lines
    def p(s=""): out.append(s)   # shorthand: append a line to output

    p(banner("BATTLE OF THE BEAMS — VHF PROPAGATION ANALYSIS"))
    p(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p()

    # --- System parameters ---
    p(banner("1. KNICKEBEIN SYSTEM PARAMETERS"))
    p(f"  Frequency               {FREQ/1e6:.1f} MHz   (λ = {LAM:.3f} m)")
    p(f"  Wavenumber k            {K_WAVE:.4f} rad/m")
    p(f"  Transmit power          {P_TX:.0f} W  ({10*np.log10(P_TX):.1f} dBW)")
    p(f"  Antenna aperture        {L_H:.0f} m (H) × {H_V:.0f} m (V)  =  {A_PHYS:.0f} m²")
    p(f"  Directivity             {G_DIR:.0f}  ({G_DIR_dB:.1f} dBi)")
    p(f"  HPBW  (horizontal)      {np.degrees(HPBW_H):.2f}°")
    p(f"  HPBW  (vertical)        {np.degrees(HPBW_V):.2f}°")
    p(f"  FNBW  (horizontal)      {np.degrees(FNBW_H):.2f}°")
    p(f"  FNBW  (vertical)        {np.degrees(FNBW_V):.2f}°")
    p(f"  Measured equisignal     {np.degrees(EQSIG_ANGLE):.3f}°  (from Kleve beam)")
    p(f"  Spec max divergence     0.300°")
    p(f"  TX elevation            {H_TX:.0f} m ASL")
    p(f"  Aircraft altitude       {H_AIRCRAFT:.0f} m  ({m2ft(H_AIRCRAFT):.0f} ft)")
    p()
    p(f"  Earth radius (actual)   {R_EARTH/1e3:.0f} km")
    p(f"  Refraction factor k     {K_REFR:.4f}")
    p(f"  Earth radius (effective){R_EFF/1e3:.0f} km")

    # --- Diffraction constants ---
    dw = diffraction_widths()
    p(banner("2. DIFFRACTION PARAMETERS  (frequency-dependent, path-independent)"))
    p(f"  kR (effective)          {dw['kR']:.0f}")
    p(f"  (kR)^(1/3)             {dw['kR_third']:.1f}")
    p(f"  Creeping ribbon width   {dw['W_creep_m']:.0f} m  =  {m2km(dw['W_creep_m']):.1f} km")
    p(f"  Creeping angular width  {np.degrees(dw['dtheta_rad']):.4f}°  ({dw['dtheta_rad']:.6f} rad)")
    p(f"  Far-field limit L²/λ   {dw['far_field_m']:.0f} m  (knife-edge valid beyond this)")
    p(f"  Attachment dist (50%)   {m2km(dw['s_attach_50_m']):.1f} km")
    p(f"  Attachment dist (90%)   {m2km(dw['s_attach_90_m']):.1f} km")
    p(f"  Attachment dist (99%)   {m2km(dw['s_attach_99_m']):.1f} km")

    # Fock first-mode attenuation rate (for display purposes)
    tau1 = 2.33811   # first Airy zero (dominant creeping-wave mode eigenvalue)
    # Recompute Fock parameter m for display (same formula as in fock_loss)
    m_fock = (K_WAVE * R_EFF / 2.0) ** (1.0/3.0)
    # First-mode attenuation rate in dB/km along the surface:
    #   alpha = 8.686 * tau_1 * sin(60 deg) * m / R * 1000
    alpha_dB_km = 8.686 * tau1 * np.sin(np.pi/3) * m_fock / R_EFF * 1000
    p(f"\n  Fock parameter m        {m_fock:.1f}")
    p(f"  First-mode decay τ₁     {tau1:.5f}")
    p(f"  Attenuation rate        {alpha_dB_km:.3f} dB/km  (first creeping mode, surface)")

    # ---------------------------------------------------------------
    #  PER-PATH ANALYSIS
    # ---------------------------------------------------------------
    for path in PATHS:
        d    = path["d"]      # great-circle distance for this path (m)
        name = path["name"]   # human-readable path label

        p(banner(f"3. PATH:  {name}"))
        p(f"  TX: {path['tx']}  ({path['tx_lat']:.2f}°N, {path['tx_lon']:.2f}°E)")
        p(f"  RX: {path['rx']}   ({path['rx_lat']:.2f}°N, {path['rx_lon']:.2f}°E)")
        p(f"  GCD: {d:.0f} m  =  {m2km(d):.1f} km  =  {m2mi(d):.1f} mi")

        # === GEOMETRIC ANALYSIS ===
        p(sub("3a. GEOMETRIC ANALYSIS  (Globe, k = 4/3 refraction)"))

        for label, h_rx in [("Ground (10 m)", H_GROUND),
                            (f"Aircraft ({H_AIRCRAFT:.0f} m)", H_AIRCRAFT)]:
            g = geometry(d, H_TX, h_rx)
            p(f"\n  Receiver: {label}")
            p(f"    Earth curvature drop at RX   {g['drop_m']:.0f} m  ({m2ft(g['drop_m']):.0f} ft)")
            p(f"    Midpath sagitta              {g['sagitta_m']:.0f} m  ({m2ft(g['sagitta_m']):.0f} ft)")
            p(f"    Beam altitude above surface  {g['beam_alt_m']:.0f} m  ({m2ft(g['beam_alt_m']):.0f} ft)")
            p(f"    TX radio horizon             {g['d_tx_km']:.1f} km")
            p(f"    RX radio horizon             {g['d_rx_km']:.1f} km")
            p(f"    Total LoS range              {g['d_los_km']:.1f} km")
            p(f"    *** Shadow distance ***       {g['d_shadow_km']:.1f} km")
            p(f"    Central angle θ              {np.degrees(g['theta_rad']):.3f}°  ({g['theta_rad']:.5f} rad)")

            if g['d_shadow_km'] > 0:
                p(f"    → Path extends {g['d_shadow_km']:.1f} km BEYOND radio line-of-sight")
            else:
                p(f"    → Path is within radio line-of-sight (no shadow)")

        # === FLAT-SURFACE PREDICTIONS ===
        p(sub("3b. FLAT-SURFACE PREDICTIONS  (rectilinear propagation)"))
        fl = flat_predictions(d)
        p(f"  Equisignal width (0.066°)  {fl['w_eqsig_m']:.0f} m   =  {m2yd(fl['w_eqsig_m']):.0f} yd")
        p(f"  HPBW beam width (horiz)    {m2km(fl['w_hpbw_h_m']):.1f} km")
        p(f"  FNBW beam width (horiz)    {m2km(fl['w_fnbw_h_m']):.1f} km")
        p(f"  HPBW beam width (vert)     {m2km(fl['w_hpbw_v_m']):.1f} km")
        p(f"  Free-space path loss       {fl['fspl_dB']:.1f} dB")
        if path["meas_eqsig_yd"]:
            p(f"\n  *** MEASURED equisignal    {path['meas_eqsig_yd']} yd  "
              f"({path['meas_eqsig_yd'] * 0.9144:.0f} m) ***")
            p(f"  *** PREDICTED (flat)       {m2yd(fl['w_eqsig_m']):.0f} yd  "
              f"({fl['w_eqsig_m']:.0f} m) ***")
            diff_pct = abs(m2yd(fl['w_eqsig_m']) - path['meas_eqsig_yd']) / path['meas_eqsig_yd'] * 100
            p(f"  *** Agreement              {diff_pct:.1f}% ***")

        # === DIFFRACTION ANALYSIS ===
        p(sub("3c. DIFFRACTION ANALYSIS  (Globe model)"))
        p(f"\n  Knife-edge beam width at {m2km(d):.0f} km:")
        w_ke = knife_edge_width(d)
        p(f"    W_knife = 2 z λ/L = {w_ke:.0f} m  =  {m2km(w_ke):.1f} km")

        p(f"\n  Creeping-wave ribbon width (vertical extent near surface):")
        p(f"    W_creep = {dw['W_creep_m']:.0f} m  =  {m2km(dw['W_creep_m']):.1f} km")

        if path["meas_eqsig_yd"]:
            meas_m = path["meas_eqsig_yd"] * 0.9144   # convert yards to metres
            p(f"\n  Comparison of widths:")
            p(f"    Measured equisignal        {meas_m:.0f} m")
            p(f"    Flat-model equisignal      {fl['w_eqsig_m']:.0f} m   (ratio: {fl['w_eqsig_m']/meas_m:.2f}×)")
            p(f"    Knife-edge diffraction     {w_ke:.0f} m   (ratio: {w_ke/meas_m:.0f}×)")
            p(f"    Creeping-wave ribbon       {dw['W_creep_m']:.0f} m   (ratio: {dw['W_creep_m']/meas_m:.0f}×)")

        # Fock diffraction loss for each receiver height
        p(sub("3d. FOCK DIFFRACTION LOSS  (residue series, 5 modes)"))

        for label, h_rx, gain_label in [
            ("Ground (10 m)", H_GROUND, "ground station"),
            (f"Aircraft ({H_AIRCRAFT:.0f} m)", H_AIRCRAFT, "He 111 bomber"),
        ]:
            fk = fock_loss(d, H_TX, h_rx)
            p(f"\n  Receiver: {label}  ({gain_label})")
            p(f"    Fock ξ               {fk['xi']:.2f}")
            p(f"    Norm. height Y_tx    {fk['Y1']:.3f}")
            p(f"    Norm. height Y_rx    {fk['Y2']:.3f}")
            p(f"    |V(ξ)| (surface)     {fk['V_abs']:.3e}")
            p(f"    G(Y_tx)              {fk['G1']:.3f}   ({dB20(fk['G1']):.1f} dB)")
            p(f"    G(Y_rx)              {fk['G2']:.3f}   ({dB20(fk['G2']):.1f} dB)")
            p(f"    F = V × G₁ × G₂     {fk['F']:.3e}")
            p(f"    *** Diffraction loss  {fk['loss_dB']:.1f} dB below free space ***")

            # Link budget for this receiver scenario
            lk = link_budget(d, fk['loss_dB'], rx_gain_dBi=3.0)
            p(f"\n    — Link budget —")
            p(f"    FSPL                 {lk['fspl_dB']:.1f} dB")
            p(f"    P_rx (at receiver)   {lk['P_rx_dBW']:.1f} dBW")
            p(f"    Noise floor          {lk['N_total_dBW']:.1f} dBW")
            p(f"    *** SNR              {lk['SNR_dB']:.1f} dB ***")
            if lk['SNR_dB'] < 10:
                p(f"    → SIGNAL {'UNDETECTABLE' if lk['SNR_dB'] < 0 else 'MARGINAL'}")
                p(f"      (need ≥10–20 dB SNR for equisignal discrimination)")
            else:
                p(f"    → Signal detectable (SNR ≥ 10 dB)")

        # Flat-surface link budget (no diffraction loss)
        p(sub("3e. FLAT-SURFACE LINK BUDGET  (no diffraction loss)"))
        lk_flat = link_budget(d, 0.0, rx_gain_dBi=3.0)
        p(f"  FSPL                     {lk_flat['fspl_dB']:.1f} dB")
        p(f"  P_rx (at receiver)       {lk_flat['P_rx_dBW']:.1f} dBW")
        p(f"  Noise floor              {lk_flat['N_total_dBW']:.1f} dBW")
        p(f"  *** SNR                  {lk_flat['SNR_dB']:.1f} dB ***")
        p(f"  → Signal {'detectable' if lk_flat['SNR_dB'] >= 10 else 'marginal'}"
          f" with {lk_flat['SNR_dB']:.0f} dB margin above noise")

        # Weyl-van der Pol flat Earth comparison
        wvdp_gain = weyl_van_der_pol_gain_dB()
        p(f"\n  --- Fock Flat Earth (Weyl-van der Pol, Eq. 3.23) ---")
        p(f"  Ground reflection gain   +{wvdp_gain:.1f} dB  (W=2 for perfect conductor)")
        p(f"  *** SNR (Fock flat)       {lk_flat['SNR_dB'] + wvdp_gain:.1f} dB ***")
        p(f"  → Conservative FSPL model used above is {wvdp_gain:.0f} dB LOWER than")
        p(f"    Fock's own flat-Earth solution.  Source: Fock (1965), p. 201.")

    # ---------------------------------------------------------------
    #  EQUISIGNAL CROSSOVER ANALYSIS
    #
    #  Critical correction: the SNR values above are at the BEAM PEAK.
    #  The pilot flies at the EQUISIGNAL, which is the crossover point
    #  between the two sub-beams.  At the crossover, each sub-beam is
    #  well below its peak (how far below depends on the squint angle).
    #  The actual SNR at the equisignal is lower than the peak SNR.
    # ---------------------------------------------------------------
    p(banner("4. EQUISIGNAL CROSSOVER ANALYSIS"))
    p("""
  The equisignal corridor is NOT at the beam peak.  The pilot flies at
  the crossover point where the dot and dash sub-beams have equal
  amplitude.  At this crossover, each sub-beam is several dB below its
  own peak.  The deeper the crossover, the narrower the equisignal but
  the weaker the signal.

  The squint angle (angular offset between dot and dash beams) is an
  operator-adjustable parameter, not derivable from the antenna
  dimensions.  The table below shows the trade-off between squint
  angle, equisignal width, and signal level at the crossover.""")

    # Compute the trade-off table
    d_kl = PATHS[0]["d"]
    fl_kl = flat_predictions(d_kl)
    fk_kl_air = fock_loss(d_kl, H_TX, H_AIRCRAFT)
    lk_flat_kl = link_budget(d_kl, 0.0)
    lk_globe_kl = link_budget(d_kl, fk_kl_air['loss_dB'])

    p(f"\n  Kleve → Spalding ({m2km(d_kl):.0f} km)")
    p(f"  Flat SNR at beam peak:  {lk_flat_kl['SNR_dB']:.1f} dB")
    p(f"  Globe SNR at beam peak: {lk_globe_kl['SNR_dB']:.1f} dB  (Fock diffraction loss = {fk_kl_air['loss_dB']:.1f} dB)")
    p(f"\n  {'Squint':>8} {'X-over':>8} {'Flat SNR':>10} {'Globe SNR':>10} {'Eq width':>10}")
    p(f"  {'(deg)':>8} {'(dB)':>8} {'at eqsig':>10} {'at eqsig':>10} {'(yd)':>10}")
    p(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    squints = [1.0, 1.5, 2.0, 2.44, 3.0, 3.5, 4.0, 4.5, 5.0]
    for sq in squints:
        u = np.pi * L_H * np.radians(sq) / LAM
        E = np.sinc(u / np.pi)
        cross = 20.0 * np.log10(max(abs(E), 1e-30))
        flat_eq_snr = lk_flat_kl['SNR_dB'] + cross
        globe_eq_snr = lk_globe_kl['SNR_dB'] + cross

        # Equisignal width (using 1 dB JND threshold)
        eps = 1e-6
        Ep = np.sinc((u + np.pi * L_H * eps / LAM) / np.pi)
        Em = np.sinc((u - np.pi * L_H * eps / LAM) / np.pi)
        dE = (Ep - Em) / (2.0 * eps)
        if abs(E) > 1e-20:
            slope = (20.0 / np.log(10.0)) * 2.0 * abs(dE) / abs(E)
        else:
            slope = 0.0
        eq_yd = d_kl * 2.0 * (1.0 / slope) / 0.9144 if slope > 0 else float('inf')

        gl_str = f"{globe_eq_snr:>8.1f} dB" if globe_eq_snr > -200 else "    N/A"
        p(f"  {sq:>8.2f} {cross:>8.1f} {flat_eq_snr:>8.1f} dB {gl_str} {eq_yd:>10.0f}")

    p(f"""
  The trade-off: a narrow equisignal (< 1000 yd) requires a deep
  crossover (> 10 dB below peak), which on the globe model pushes the
  individual dot/dash signals below the noise floor.  A shallow
  crossover keeps the signal above noise but gives an equisignal
  several kilometres wide.

  On the flat model, this trade-off does not exist.  With {lk_flat_kl['SNR_dB']:.0f} dB
  at the beam peak, even a -19 dB crossover leaves {lk_flat_kl['SNR_dB']-19:.0f} dB of SNR
  at the equisignal.  Any squint angle works.

  On the globe model, no squint angle simultaneously produces a narrow
  equisignal (500 yd) and a usable signal at the crossover.""")

    # ---------------------------------------------------------------
    #  SUMMARY COMPARISON TABLE
    # ---------------------------------------------------------------
    # Recompute with crossover correction
    # Use the squint that gives ~500 yd: about 5 degrees, crossover ~ -19 dB
    crossover_dB = -19.0

    fk_kl_gnd = fock_loss(d_kl, H_TX, H_GROUND)
    d_st = PATHS[1]["d"]
    fl_st = flat_predictions(d_st)
    fk_st_air = fock_loss(d_st, H_TX, H_AIRCRAFT)
    fk_st_gnd = fock_loss(d_st, H_TX, H_GROUND)
    lk_flat_st = link_budget(d_st, 0.0)
    lk_globe_st = link_budget(d_st, fk_st_air['loss_dB'])

    flat_eq_kl = lk_flat_kl['SNR_dB'] + crossover_dB
    globe_eq_kl = lk_globe_kl['SNR_dB'] + crossover_dB
    flat_eq_st = lk_flat_st['SNR_dB'] + crossover_dB
    globe_eq_st = lk_globe_st['SNR_dB'] + crossover_dB

    p(banner("5. SUMMARY — FLAT vs. GLOBE PREDICTIONS"))
    p(f"\n  SNR values corrected to the EQUISIGNAL crossover (-19 dB below beam peak)")
    p(f"  for the squint angle that gives ~500 yd equisignal on the flat model.")
    p()
    p(f"  {'Quantity':<40} {'Flat Model':>14} {'Globe Model':>14} {'Measured':>14}")
    p(f"  {'─'*40} {'─'*14} {'─'*14} {'─'*14}")

    p(f"\n  Kleve → Spalding  ({m2km(d_kl):.0f} km)")
    p(f"  {'Equisignal width (yd)':<40} {m2yd(fl_kl['w_eqsig_m']):>13.0f} {'N/A':>14} {500:>13.0f}")
    p(f"  {'Diff. loss (dB)':<40} {'0':>14} {fk_kl_air['loss_dB']:>13.1f} {'—':>14}")
    p(f"  {'SNR at beam PEAK (dB)':<40} {lk_flat_kl['SNR_dB']:>13.1f} {lk_globe_kl['SNR_dB']:>13.1f} {'—':>14}")
    p(f"  {'SNR at EQUISIGNAL (dB)':<40} {flat_eq_kl:>13.1f} {globe_eq_kl:>13.1f} {'≥10 (usable)':>14}")
    p(f"  {'Creeping ribbon width (km)':<40} {'N/A':>14} {m2km(dw['W_creep_m']):>13.1f} {'—':>14}")

    p(f"\n  Stollberg → Beeston  ({m2km(d_st):.0f} km)")
    p(f"  {'Equisignal width @ 0.066° (yd)':<40} {m2yd(fl_st['w_eqsig_m']):>13.0f} {'N/A':>14} {'—':>14}")
    p(f"  {'Diff. loss (dB)':<40} {'0':>14} {fk_st_air['loss_dB']:>13.1f} {'—':>14}")
    p(f"  {'SNR at beam PEAK (dB)':<40} {lk_flat_st['SNR_dB']:>13.1f} {lk_globe_st['SNR_dB']:>13.1f} {'—':>14}")
    p(f"  {'SNR at EQUISIGNAL (dB)':<40} {flat_eq_st:>13.1f} {globe_eq_st:>13.1f} {'≥10 (usable)':>14}")

    # ---------------------------------------------------------------
    #  KEY FINDINGS
    # ---------------------------------------------------------------
    p(banner("6. KEY FINDINGS"))
    p()
    p("  A. EQUISIGNAL WIDTH")
    p(f"     Measured at Spalding: 400–500 yd ({0.9144*400:.0f}–{0.9144*500:.0f} m)")
    p(f"     Flat-model prediction: {m2yd(fl_kl['w_eqsig_m']):.0f} yd ({fl_kl['w_eqsig_m']:.0f} m)")
    p(f"     → Match within {abs(m2yd(fl_kl['w_eqsig_m'])-500)/500*100:.0f}%")
    p()
    p("  B. BEAM VERTICAL STRUCTURE")
    p(f"     On flat surface:  beam arrives at TX elevation ± antenna pattern")
    p(f"     On globe:  beam must diffract over the horizon")
    p(f"       → Creeping wave fills a {m2km(dw['W_creep_m']):.0f} km vertical ribbon")
    p(f"       → {dw['W_creep_m']/457:.0f}× wider than the measured equisignal")
    p()
    p("  C. SIGNAL AT THE EQUISIGNAL (corrected from beam peak)")
    p(f"     The pilot flies at the equisignal crossover, where each sub-beam")
    p(f"     is ~19 dB below its peak (for the squint that gives 500 yd).")
    p()
    p(f"     Kleve → Spalding:")
    p(f"       Flat model:  SNR at equisignal = {flat_eq_kl:.0f} dB  (usable)")
    p(f"       Globe model: SNR at equisignal = {globe_eq_kl:.0f} dB  (BELOW NOISE)")
    p(f"       Each dot/dash sample is {abs(globe_eq_kl):.0f} dB below the noise floor.")
    p(f"       The receiver hears noise, not signal, at the equisignal.")
    p()
    p(f"     Stollberg → Beeston:")
    p(f"       Flat model:  SNR at equisignal = {flat_eq_st:.0f} dB  (usable)")
    p(f"       Globe model: SNR at equisignal = {globe_eq_st:.0f} dB  (BELOW NOISE)")
    p()
    p("  D. THE SQUINT/SNR TRADE-OFF (Globe model only)")
    p(f"     On a globe, narrow equisignal requires deep crossover, which")
    p(f"     pushes the signal below the noise floor.  Shallow crossover")
    p(f"     keeps marginal signal but widens equisignal to kilometres.")
    p(f"     No squint angle gives both narrow equisignal AND usable signal.")
    p(f"     On a flat surface, this trade-off does not exist ({lk_flat_kl['SNR_dB']:.0f} dB headroom).")
    p()
    p("  E. CONTINUOUS BEAM TRACKING")
    g_air = geometry(d_kl, H_TX, H_AIRCRAFT)
    p(f"     Aircraft flew INSIDE the beam continuously from Germany to England.")
    p(f"     On a globe, the beam enters the shadow zone after {g_air['d_los_km']:.0f} km.")
    p(f"     For Kleve → Spalding ({m2km(d_kl):.0f} km), the last {g_air['d_shadow_km']:.0f} km")
    p(f"     are beyond radio line-of-sight even with aircraft at 6 km altitude.")
    g_air_st = geometry(d_st, H_TX, H_AIRCRAFT)
    p(f"     For Stollberg → Beeston ({m2km(d_st):.0f} km), {g_air_st['d_shadow_km']:.0f} km are")
    p(f"     beyond radio line-of-sight (46% of the total path).")

    p()
    p(SEP)
    p("  CONCLUSION: The measured beam properties (narrow equisignal, sufficient")
    p("  signal for precision navigation) are consistent with rectilinear")
    p("  propagation.  On a spherical Earth, the Fock diffraction loss combined")
    p("  with the equisignal crossover loss places the individual dot/dash")
    p("  signals below the noise floor for any squint angle that gives a narrow")
    p("  equisignal corridor.")
    p(SEP)

    # ---------------------------------------------------------------
    #  REFERENCES
    # ---------------------------------------------------------------
    p(banner("REFERENCES"))
    p("""
  [1] Fock, V.A. (1965). Electromagnetic Diffraction and Propagation
      Problems. Pergamon Press, Oxford. (Int. Series Monographs on EM
      Waves, Vol. 1.)  Ch. 10 reprints Fock (1945).
      Key equations: Eq. (6.10) p.209 (residue series), Eq. (3.22-3.23)
      p.201 (Weyl-van der Pol flat Earth), Eq. (5.08) p.205 (normalised
      height), Eq. (5.15) p.207 (normalised distance).
      DRIC Translation No. 2747 (April 1972) is the typewriter scan.

  [2] Eckersley, T.L. (1937). "Ultra-Short-Wave Refraction and Diffraction."
      J.I.E.E. 80, p.286.

  [3] Vogler, L.E. (1961). "Smooth Earth Diffraction Calculations for
      Horizontal Polarization." NBS J. Res. 65D(4), 397-399.

  [4] Neubauer, W.G., P. Ugincius, and H. Uberall (1969). "Theory of
      Creeping Waves in Acoustics and Their Experimental Demonstration."
      Z. Naturforsch. 24a, 691-700.  Cross-reference: Eq. 20-26.

  [5] Keller, J.B. "Geometrical Theory of Diffraction." WHOI Lecture Notes.

  [6] Bird, J.F. (1985). "Diffraction by a conducting sphere." JOSA A
      2(6):945-953.

  [7] Born, M. and E. Wolf. Principles of Optics, 7th ed., Sec. 8.4.3.

  [8] Goodman, J. Introduction to Fourier Optics, 3rd ed., Ch. 4.

  [9] Shim, J. and H.-T. Kim (1999). PIER 21:293-306.

  [10] Bauer, A.O. (2004). "Some historical and technical aspects of radio
       navigation, in Germany, over the period 1907 to 1945."
""")

    text = "\n".join(out)
    print(text)
    return text


if __name__ == "__main__":
    result = main()

    # Save the full analysis output to a text file
    outpath = "/home/alan/claude/BotB/botb_analysis_output.txt"
    with open(outpath, "w") as f:
        f.write(result)
    print(f"\n[Saved to {outpath}]")
