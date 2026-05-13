"""Shared parameters and helpers ported verbatim from
ITU_CERTIFIED_Battle_of_the_Beams_Calc2_v9_1_1.xlsx.

Numbers and formulas come from:
- const sheet  ->  physical constants
- ground sheet ->  conductivity / permittivity by ground type
- stations sheet -> Knickebein TX towers
- targets sheet -> RX targets (Midlands cities, Telefunken sea-tests)
- paths_refs sheet -> path geometry table
- ITU sheet rows 2-3 -> ITU-R P.526-16 Fock GE columns
- ITU sheet rows 6-7 -> Sommerfeld-Norton FE columns
- Squint Sandbox sheet -> editable squint/aperture defaults

Defined names map directly to symbol names below (kl_freq_mhz etc).
"""
from __future__ import annotations
import math
import cmath


# --- const sheet ---
EPS_0 = 8.8541878128e-12          # F/m
MU_0  = 1.25663706212e-6          # H/m
C_VAC = 1.0 / math.sqrt(EPS_0 * MU_0)
K_BOLTZ = 1.380649e-23            # J/K
R_EARTH = 6_371_000.0             # m
K_REFRAC = 4.0 / 3.0              # ITU-R P.453 4/3 effective Earth
R_EFFECTIVE = K_REFRAC * R_EARTH  # 8.495e6 m
T_SYS = 290.0                     # K
RX_NF_DB = 10.0                   # dB
RX_BW_HZ = 500.0                  # Hz (MCW matched filter)
ETA_0 = 376.730313                # ohm, free space impedance


# --- ground sheet ---
GROUND = {
    "land": dict(sigma=5e-3,  eps_r=15, note="ITU-R P.527 class 3 typical"),
    "sea":  dict(sigma=5.0,   eps_r=70, note="warm/typical North Sea"),
    "wet":  dict(sigma=2e-2,  eps_r=30, note="wet soil"),
    "dry":  dict(sigma=1e-3,  eps_r=4,  note="dry soil"),
}


# --- stations sheet ---
STATIONS = {
    "Kleve": dict(
        short_id="Kn-4", lat_deg=51.7886, lon_deg=6.1031,
        terrain_elev_m=83, frame_height_m=28, h_tx_m=111,
        W_m=99, H_m=29, freq_MHz=31.5, pol="vertical",
        Ptx_W=3000, ground="land", squint_deg=5,
    ),
    "Stollberg": dict(
        short_id="Kn-2", lat_deg=54.6436, lon_deg=8.9447,
        terrain_elev_m=44, frame_height_m=28, h_tx_m=72,
        W_m=99, H_m=29, freq_MHz=31.5, pol="vertical",
        Ptx_W=3000, ground="sea", squint_deg=5,
    ),
    "Greny": dict(
        short_id="Kn-7", lat_deg=49.9467, lon_deg=1.29,
        terrain_elev_m=134, frame_height_m=28, h_tx_m=162,
        W_m=99, H_m=29, freq_MHz=31.5, pol="vertical",
        Ptx_W=3000, ground="sea", squint_deg=5,
    ),
    "Beaumont-Hague": dict(
        short_id="Kn-9", lat_deg=49.6733, lon_deg=-1.8525,
        terrain_elev_m=169, frame_height_m=28, h_tx_m=197,
        W_m=99, H_m=29, freq_MHz=31.5, pol="vertical",
        Ptx_W=3000, ground="sea", squint_deg=5,
    ),
}


# --- targets sheet ---
TARGETS = {
    "Spalding":   dict(lat=52.787,   lon=-0.153,  rx_alt_m=6000, note="Bufton 21 Jun 1940 Kleve beam"),
    "Beeston":    dict(lat=52.927,   lon=-1.215,  rx_alt_m=6000, note="Bufton 21 Jun 1940 Stollberg beam"),
    "Derby":      dict(lat=52.922,   lon=-1.475,  rx_alt_m=6000, note="Operational target Rolls-Royce"),
    "Birmingham": dict(lat=52.4862,  lon=-1.8904, rx_alt_m=6000, note="Operational target"),
    "Retford":    dict(lat=53.4,     lon=-1.0,    rx_alt_m=6000, note="Enigma intercept 5 Jun 1940"),
    "London":     dict(lat=51.5074,  lon=-0.1278, rx_alt_m=6000, note="Operational target"),
    "Liverpool":  dict(lat=53.4084,  lon=-2.9916, rx_alt_m=6000, note="Stollberg target"),
    "Cardiff":    dict(lat=51.4816,  lon=-3.1791, rx_alt_m=6000, note="Beaumont-Hague target"),
    "Plymouth":   dict(lat=50.3755,  lon=-4.1427, rx_alt_m=6000, note="Beaumont-Hague target"),
    "TF 400 km":  dict(lat=54.589878, lon=2.518619,  rx_alt_m=4000, note="Telefunken Sep 1939 FuBl1 rod"),
    "TF 500 km":  dict(lat=54.514617, lon=1.112564,  rx_alt_m=4000, note="Telefunken Sep 1939 FuBl1 wire"),
    "TF 700 km":  dict(lat=54.342286, lon=-1.932556, rx_alt_m=4000, note="Telefunken Sep 1939 regen rod"),
    "TF 800 km":  dict(lat=54.211456, lon=-3.499303, rx_alt_m=4000, note="Telefunken Sep 1939 regen wire"),
    "TF 1000 km": dict(lat=53.931886, lon=-6.446086, rx_alt_m=4000, note="Telefunken Sep 1939 special dipole"),
}


# --- Squint Sandbox defaults (row 4, row 9, row 18) ---
SANDBOX = dict(
    squint_deg=5.0,    # B4 / C4
    W_m=99.0,          # B9 / C9
    H_m=20.0,          # B18 / C18 (Squint Sandbox aperture H input)
)


# =====================================================================
# Geometry
# =====================================================================
def great_circle_m(lat1, lon1, lat2, lon2, R=R_EARTH):
    """Haversine great-circle distance (ITU O column formula)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def freq_to_wavelen(f_MHz):
    return C_VAC / (f_MHz * 1e6)


def wavenumber(f_MHz):
    return 2 * math.pi / freq_to_wavelen(f_MHz)


# =====================================================================
# ITU-R P.526-16 Fock smooth-Earth diffraction (ITU sheet cols V-AR)
# =====================================================================
def p526_beta_and_K(ground, pol, f_MHz, k_refrac=K_REFRAC):
    """beta polarisation parameter and K via Eq.16, 16a.
    Returns (beta_final, K_for_floor)."""
    sigma = GROUND[ground]["sigma"]
    f_cut_MHz = 20.0 if ground == "land" else 300.0
    beta1 = (pol == "horizontal") or (f_MHz > f_cut_MHz)
    K2 = 6.89 * sigma / (k_refrac**(2/3) * f_MHz**(5/3))    # Eq. 16a
    K  = math.sqrt(K2)
    beta_eq16 = (1 + 1.6*K2 + 0.67*K2**2) / (1 + 4.5*K2 + 1.53*K2**2)
    if beta1:
        return 1.0, 0.0
    return beta_eq16, K


def p526_X(d_m, beta, f_MHz, a_e=R_EFFECTIVE):
    lam = freq_to_wavelen(f_MHz)
    return beta * (math.pi / (lam * a_e**2))**(1/3) * d_m


def p526_Y(h_m, beta, f_MHz, a_e=R_EFFECTIVE):
    lam = freq_to_wavelen(f_MHz)
    return 2 * beta * (math.pi**2 / (lam**2 * a_e))**(1/3) * h_m


def F_of_X(X):
    """Eq. 14 / 15 — long-path vs short-path branch."""
    if X >= 1.6:
        return 11 + 10*math.log10(X) - 17.6*X
    X = max(X, 1e-10)
    return -20*math.log10(X) - 5.6488 * X**1.425


def G_of_Y(B, K_for_floor=0.0):
    """Eq. 17, with Eq. 18 floor when K_for_floor > 0."""
    if B > 2:
        raw = 17.6*math.sqrt(B - 1.1) - 5*math.log10(B - 1.1) - 8
    else:
        Bp = max(B, 1e-10)
        raw = 20*math.log10(Bp + 0.1 * Bp**3)
    if K_for_floor > 0:
        floor = 2 + 20*math.log10(K_for_floor)
        return max(raw, floor)
    return raw


def fock_diff_loss_dB(d_m, h_tx, h_rx, f_MHz, ground, pol, a_e=R_EFFECTIVE):
    """Total diffraction loss (positive dB) past line of sight,
    zero inside the radio horizon."""
    d_los_tx = math.sqrt(2 * a_e * h_tx)
    d_los_rx = math.sqrt(2 * a_e * h_rx)
    if d_m < d_los_tx + d_los_rx:
        return 0.0
    beta, K_floor = p526_beta_and_K(ground, pol, f_MHz, k_refrac=a_e/R_EARTH)
    X  = p526_X(d_m, beta, f_MHz, a_e)
    Y1 = p526_Y(h_tx, beta, f_MHz, a_e)
    Y2 = p526_Y(h_rx, beta, f_MHz, a_e)
    Fx = F_of_X(X)
    G1 = G_of_Y(beta*Y1, K_floor)
    G2 = G_of_Y(beta*Y2, K_floor)
    total = Fx + G1 + G2                       # E/E0 in dB (negative)
    return max(-total, 0.0)


# =====================================================================
# Link budget pieces (ITU cols AS-BP)
# =====================================================================
def fspl_dB(d_m, f_MHz):
    return 20*math.log10(4*math.pi*d_m / freq_to_wavelen(f_MHz))


def aperture_gain_dBi(W_m, H_m, f_MHz):
    lam = freq_to_wavelen(f_MHz)
    G_lin = 4*math.pi*W_m*H_m / lam**2
    return 10*math.log10(G_lin)


def galactic_Fa_dB(f_MHz):
    """ITU-R P.372-16 Eq. 14 below 100 MHz."""
    return 52 - 23*math.log10(f_MHz)


def noise_floor_dBW(f_MHz, rx_nf_dB=RX_NF_DB, T_sys=T_SYS, B=RX_BW_HZ):
    thermal_dBW = 10*math.log10(K_BOLTZ * T_sys * B)
    Fa = galactic_Fa_dB(f_MHz)
    return thermal_dBW + max(rx_nf_dB, Fa)


def crossover_loss_dB(W_m, theta_deg, f_MHz):
    """sinc(pi W sin(theta) / lambda) in dB.  ITU sheet const!B13."""
    lam = freq_to_wavelen(f_MHz)
    u = math.pi * W_m * math.sin(math.radians(theta_deg)) / lam
    if abs(u) < 1e-12:
        return 0.0
    return 20*math.log10(abs(math.sin(u)/u))


# =====================================================================
# Sommerfeld-Norton flat-Earth three-term Ez (ITU cols BQ-CY)
# =====================================================================
def sommerfeld_FE(d_m, h_tx, h_rx, f_MHz, ground, Ptx_W, G_tx_lin):
    """Returns dict with |Ez|, boosted E, P_rx, intermediate complex terms."""
    sigma = GROUND[ground]["sigma"]
    eps_r = GROUND[ground]["eps_r"]
    lam = freq_to_wavelen(f_MHz)
    k = 2*math.pi/lam
    x = 18000.0 * sigma / f_MHz                  # BQ
    dh1 = h_rx - h_tx                            # BR
    dh2 = h_rx + h_tx                            # BS
    r1 = math.sqrt(d_m**2 + dh1**2)              # BT
    r2 = math.sqrt(d_m**2 + dh2**2)              # BU
    cos2_psi1 = (d_m/r1)**2                      # BV
    cos2_psi2 = (d_m/r2)**2                      # BW
    sin_psi2 = dh2/r2                            # BX
    n2 = complex(eps_r, -x)                      # BZ
    u2 = 2.0/n2                                  # BY
    n2_sin_psi2 = n2 * sin_psi2                  # CA
    sqrt_term = cmath.sqrt(n2 - cos2_psi2)       # CB
    Rv = (n2_sin_psi2 - sqrt_term)/(n2_sin_psi2 + sqrt_term)  # CC
    one_minus_Rv = 1 - Rv                        # CD
    one_minus_u2_cos2 = 1 - u2*cos2_psi2         # CE
    minus_j_2kr2 = complex(0.0, -2*k*r2)         # CF
    w_num = minus_j_2kr2 * u2 * one_minus_u2_cos2  # CG
    w = w_num / one_minus_Rv                     # CH
    two_w = 2*w                                  # CI
    F = (-1/two_w) + (-3/two_w**2) + (-15/two_w**3) + (-105/two_w**4)  # CJ
    exp_jkr1 = cmath.exp(complex(0.0, -k*r1))    # CK
    exp_jkr2 = cmath.exp(complex(0.0, -k*r2))    # CL
    direct  = cos2_psi1 * exp_jkr1 / r1          # CM
    reflect = cos2_psi2 * Rv * exp_jkr2 / r2     # CN
    u4 = u2**2                                   # CO
    bracket = 1 - u2 + u4*cos2_psi2              # CP
    surface = one_minus_Rv * bracket * F * exp_jkr2 / r2  # CQ
    Ez_sum = direct + reflect + surface          # CR
    abs_sum = abs(Ez_sum)                        # CS
    sqrt_90P = math.sqrt(90*Ptx_W)               # CT
    Ez_Vpm = sqrt_90P * abs_sum                  # CU
    sqrt_G_over_1p5 = math.sqrt(G_tx_lin / 1.5)  # CV
    E_boosted = Ez_Vpm * sqrt_G_over_1p5         # CW
    # P_rx with rx_gain = 0 dBi -> 10^(0/10) = 1
    P_rx_W = E_boosted**2 * lam**2 / (8*math.pi*ETA_0)
    return dict(
        x=x, r1=r1, r2=r2, Rv=Rv, w=w, F=F,
        direct=direct, reflect=reflect, surface=surface,
        Ez_sum=Ez_sum, abs_sum=abs_sum, Ez_Vpm=Ez_Vpm,
        E_boosted=E_boosted, P_rx_W=P_rx_W,
        P_rx_dBW=10*math.log10(P_rx_W),
    )


# =====================================================================
# Convenience wrappers
# =====================================================================
def voltage_50ohm_uV(P_dBW):
    """dBμV = dBW + 137; μV = 10^(dBμV/20)."""
    return 10 ** ((P_dBW + 137)/20)


def equisignal_corridor_width_m(d_m, W_m, theta_deg, f_MHz):
    """Main!D2 / Main!E2 -> full equisignal corridor width at target range."""
    lam = freq_to_wavelen(f_MHz)
    u  = math.pi * W_m * math.sin(math.radians(theta_deg)) / lam
    uc = math.pi * W_m * math.cos(math.radians(theta_deg)) / lam
    one_over_20 = 10**(1/20)
    num = 2 * d_m * math.sin(u) * u * (one_over_20 - 1)
    den = (math.sin(u) - u*math.cos(u)) * uc * (one_over_20 + 1)
    return num / den


def link_budget(tx_station, target_name, model="fock"):
    """One-shot solve mirroring ITU row 2 (KL Fock), row 3 (ST Fock),
    row 6 (KL SN), row 7 (ST SN). Returns a dict.
    model in {'fock','sommerfeld'}."""
    s = STATIONS[tx_station]
    t = TARGETS[target_name]
    f_MHz = s["freq_MHz"]
    d_m = great_circle_m(s["lat_deg"], s["lon_deg"], t["lat"], t["lon"])
    h_tx = s["h_tx_m"]
    h_rx = t["rx_alt_m"]
    lam  = freq_to_wavelen(f_MHz)
    ground = "sea" if target_name.startswith("TF") else s["ground"]
    G_tx_dBi = aperture_gain_dBi(s["W_m"], s["H_m"], f_MHz)
    G_tx_lin = 10**(G_tx_dBi/10)
    P_tx_dBW = 10*math.log10(s["Ptx_W"])
    FSPL = fspl_dB(d_m, f_MHz)
    N_dBW = noise_floor_dBW(f_MHz)
    cross_dB = crossover_loss_dB(s["W_m"], s["squint_deg"], f_MHz)

    if model == "fock":
        L_dB = fock_diff_loss_dB(d_m, h_tx, h_rx, f_MHz, ground, s["pol"])
        P_rx_dBW = P_tx_dBW + G_tx_dBi - FSPL - L_dB
    elif model == "sommerfeld":
        res = sommerfeld_FE(d_m, h_tx, h_rx, f_MHz, ground, s["Ptx_W"], G_tx_lin)
        L_dB = None
        P_rx_dBW = res["P_rx_dBW"]
    else:
        raise ValueError(model)

    SNR_peak = P_rx_dBW - N_dBW
    SNR_eq   = SNR_peak + cross_dB
    V_eq_uV  = voltage_50ohm_uV(P_rx_dBW + cross_dB)
    V_noise_uV = voltage_50ohm_uV(N_dBW)
    return dict(
        tx=tx_station, target=target_name, model=model, ground=ground,
        d_m=d_m, d_km=d_m/1000,
        lam_m=lam, h_tx_m=h_tx, h_rx_m=h_rx,
        G_tx_dBi=G_tx_dBi, P_tx_dBW=P_tx_dBW, FSPL_dB=FSPL,
        diffraction_loss_dB=L_dB,
        P_rx_dBW=P_rx_dBW,
        N_dBW=N_dBW, SNR_peak_dB=SNR_peak,
        crossover_dB=cross_dB, SNR_eq_dB=SNR_eq,
        V_eq_uV=V_eq_uV, V_noise_uV=V_noise_uV,
    )
