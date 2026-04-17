"""
Generate comparison graphs of ITU-R P.526-16 (Fock smooth-Earth diffraction)
vs ITU-R P.368 GRWAVE (ground-wave propagation) for the Knickebein paths.

Both are ITU international standards. P.526 is what the BotB null doc uses;
P.368 GRWAVE is ITU's own Fortran ground-wave calculator, which is the
physics Dan Dano specifically invoked in his response video.

Outputs three graph families, matching the existing BotB style:
 1. Distance sweep curves (Kleve TX 111m, Stollberg TX 72m, Telefunken TX 72m)
 2. Master bar chart comparing all paths at both standards

Style: dark background #1a1a1a, white noise floor, faint yellow detection
floor, green = P.526, magenta = P.368 GRWAVE.

Saves PNGs to both /home/alan/claude/BotB/ and the vault Attachments folder.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

# Import grwave FIRST, before touching sys.path. The editable pip install
# of grwave points at /home/alan/claude/BotB/grwave (the cloned repo root),
# and inside that the actual Python package is grwave/grwave/ — so the
# outer `grwave` resolves to a namespace package and the real module with
# the `grwave(params)` function lives at `grwave.grwave`. We bind the inner
# package to grwave_pkg so the rest of the script can call
# grwave_pkg.grwave(params).
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_HERE)
sys.path.insert(0, _REPO_ROOT)  # parent repo root holds grwave + botb_itu_analysis

import grwave.grwave as grwave_pkg

from botb_itu_analysis import (
    itu_diffraction_loss,
    link_budget,
    line_of_sight,
    sommerfeld_norton_snr_peak,
    CROSSOVER_dB,
    N_FLOOR_dBW,
    G_DIR_dB,
    DETECT_dB,
)


BOTB_DIR = _os.path.join(_HERE, "graphs")
_os.makedirs(BOTB_DIR, exist_ok=True)
VAULT_DIR = "/home/alan/Documents/multi_2/Attachments"

# Common constants
FREQ_MHz = 31.5
TX_POWER_W = 3000          # Large Knickebein Telefunken spec

# Style
BG_COLOR = "#1a1a1a"
COL_FRIIS = "#00E676"      # bright green (Friis flat-Earth baseline, FE)
COL_SN = "#FFD600"         # gold yellow (Sommerfeld-Norton plane Earth)
COL_P526 = "#FF1493"       # magenta (Fock diffraction, existing style)
COL_P368 = "#00E5FF"       # cyan (ITU P.368 ground-wave)
COL_NOISE = "white"
COL_DETECT = "#FFEB3B"     # faint yellow (detection floor on dB plots)
COL_FUBL2 = "#FFA726"      # light orange (FuBl 2 at Dan's 2 μV estimate)
COL_TARGET = "#90C4E0"     # light blue (target distance marker)


# ================================================================
#  PATHS
# ================================================================

# Pattern: (name, distance_km, tx_alt_m, rx_alt_m, label, path_type)
# path_type controls ground profile: "land" = dry ground, "sea" = seawater
KN_PATHS = [
    # Kleve paths (predominantly overland)
    ("Kleve→Spalding",     440, 111, 6000, "Bufton 21 Jun 1940",     "land"),
    ("Kleve→Retford",      512, 111, 6000, "Enigma intercept",       "land"),
    ("Kleve→Derby",        529, 111, 6000, "Rolls-Royce Merlin",     "land"),
    ("Kleve→Birmingham",   550, 111, 6000, "",                       "land"),
    # Stollberg paths (mixed sea + land)
    ("Stollberg→Beeston",  694, 72, 6000, "Cross-beam target",       "sea"),
    ("Stollberg→Derby",    711, 72, 6000, "",                        "sea"),
    ("Stollberg→Birmingham", 754, 72, 6000, "",                      "sea"),
    ("Stollberg→Liverpool",  791, 72, 6000, "",                      "sea"),
]

TF_PATHS = [
    ("TF 400 km", 400, 72, 4000, "FuBl 1 rod",        "sea"),
    ("TF 500 km", 500, 72, 4000, "FuBl 1 wire",       "sea"),
    ("TF 700 km", 700, 72, 4000, "FuBl+sel rod",      "sea"),
    ("TF 800 km", 800, 72, 4000, "FuBl+sel wire",     "sea"),
    ("TF 1000 km", 1000, 72, 4000, "special xdip",    "sea"),
]

GROUND = {
    "sea":  {"sigma": 5.0,   "epslon": 70.0, "label": "seawater"},
    "land": {"sigma": 5e-3,  "epslon": 15.0, "label": "avg land"},
}


# ================================================================
#  ITU-R P.526 SNR calculation at arbitrary distance
# ================================================================

def sn_snr_peak(d_km, tx_m, rx_m, ground_name="land",
                freq_mhz=FREQ_MHz, rx_gain_dBi=3.0):
    """Sommerfeld-Norton plane-earth peak SNR. Wrapper around the
    library function for consistency with the p526/p368 helpers."""
    return sommerfeld_norton_snr_peak(
        d_km, tx_m, rx_m, ground=ground_name,
        freq=freq_mhz * 1e6, rx_gain_dBi=rx_gain_dBi)


def p526_snr_peak(d_km, tx_m, rx_m, ground_name="land",
                  freq_mhz=FREQ_MHz, rx_gain=0.0):
    """ITU-R P.526 globe-model peak SNR (no equisignal crossover subtracted)
    at a single distance, using the same parameters as botb_itu_analysis.

    ground_name selects the P.526 β / K(Eq. 16) path: "land" for overland
    paths (Kleve), "sea" for overwater paths (Stollberg, Greny, Telefunken).
    At 31.5 MHz vert pol this is a no-op for "land" (β=1 per the 20 MHz
    cut) but it is load-bearing for "sea" (β≈0.81, Eq. 18 G(Y) clamp)."""
    d_m = d_km * 1000.0
    freq_hz = freq_mhz * 1e6
    itu = itu_diffraction_loss(d_m, tx_m, rx_m, freq_hz,
                                ground=ground_name)
    lb = link_budget(d_m, itu["loss_dB"], freq_hz, rx_gain)
    return lb["SNR_dB"]


# ================================================================
#  ITU-R P.368 GRWAVE SNR calculation at arbitrary distance
# ================================================================

# Convert grwave field strength (dBuV/m) to SNR (dB above receiver noise
# floor) using the canonical E->P conversion:
#   P_rx (dBW) = E (dBuV/m) - 120 - 31.7 + 169.5 - 20*log10(f_Hz) + G_r
# at f = 31.5 MHz with isotropic RX (G_r = 0 dBi):
#   P_rx (dBW) = E_dBuVm - 132.16
# SNR (dB above N_FLOOR_dBW = -151.2) = E_dBuVm + (-132.16 - N_FLOOR_dBW)
#                                     = E_dBuVm + 19.04  (isotropic RX)
#
# grwave uses isotropic TX. Knickebein uses a directional array with G_tx =
# 26 dBi. To compare to P.526 peak SNR (which includes the 26 dBi TX gain),
# we add G_DIR_dB to the grwave-derived SNR.

_E_TO_SNR_CONST = 19.04   # E_dBuVm + this = SNR (isotropic TX, isotropic RX)


def p368_snr_peak(d_km, tx_m, rx_m, ground_name,
                  freq_mhz=FREQ_MHz, tx_gain=G_DIR_dB):
    """ITU-R P.368 globe-model peak SNR via grwave Fortran at a single
    distance. Returns peak SNR in dB including the Knickebein TX directivity
    (26 dBi) and isotropic RX."""
    g = GROUND[ground_name]
    # Run grwave from well before the target out to just past it so we hit a
    # sample point near d_km. grwave has a minimum DMIN and a coarser step
    # than we want, so we just set a range that definitely includes d_km.
    d_max = max(d_km + 100, 200)
    d_step = max(10, d_km // 20)   # ~20 samples in the range
    params = {
        "freqMHz": freq_mhz,
        "sigma": g["sigma"],
        "epslon": g["epslon"],
        "dmax": d_max,
        "hrr": rx_m,
        "htt": tx_m,
        "dstep": d_step,
        "txwatt": TX_POWER_W,
    }
    data = grwave_pkg.grwave(params)
    if len(data) == 0:
        return float("nan")
    distances = data.index.values.astype(float)
    # Interpolate in dB space to the requested distance
    fs_vals = data["fs"].values.astype(float)
    # Linear interpolation in distance
    if d_km <= distances.min():
        fs_dBuVm = fs_vals[0]
    elif d_km >= distances.max():
        fs_dBuVm = fs_vals[-1]
    else:
        fs_dBuVm = float(np.interp(d_km, distances, fs_vals))
    # Convert to SNR (adding TX gain, assuming isotropic RX)
    snr = fs_dBuVm + _E_TO_SNR_CONST + tx_gain
    return snr


def p368_sweep(tx_m, rx_m, ground_name, d_min_km, d_max_km, d_step_km,
               freq_mhz=FREQ_MHz, tx_gain=G_DIR_dB):
    """One-shot grwave sweep over a distance range. Returns (distances_km,
    snr_dB) arrays."""
    g = GROUND[ground_name]
    params = {
        "freqMHz": freq_mhz,
        "sigma": g["sigma"],
        "epslon": g["epslon"],
        "dmax": d_max_km,
        "hrr": rx_m,
        "htt": tx_m,
        "dstep": d_step_km,
        "txwatt": TX_POWER_W,
    }
    data = grwave_pkg.grwave(params)
    d = data.index.values.astype(float)
    fs = data["fs"].values.astype(float)
    snr = fs + _E_TO_SNR_CONST + tx_gain
    # Restrict to [d_min, d_max]
    mask = (d >= d_min_km) & (d <= d_max_km)
    return d[mask], snr[mask]


# ================================================================
#  PLOTTING
# ================================================================

def save(fig, fname):
    """Save a figure to both the BotB dir and the vault Attachments."""
    for outdir in (BOTB_DIR, VAULT_DIR):
        outpath = os.path.join(outdir, fname)
        fig.savefig(outpath, dpi=200, bbox_inches="tight",
                    facecolor=BG_COLOR)
        print(f"  Saved: {outpath}")
    plt.close(fig)


def plot_station_sweep(station_name, tx_m, rx_m, ground_name, freq_mhz,
                       d_max_km, targets, fname, title_suffix="",
                       y_mode="snr", receiver="s27"):
    """Distance sweep graph comparing peak SNR/V_rx for the Friis flat,
    ITU P.526 Fock, and ITU P.368 GRWAVE models.

    y_mode:
        "snr" (default): peak SNR above receiver noise floor in dB.
        "uv":  peak voltage at the receiver's 50 Ω input in linear μV
               on a log y axis.
    receiver (only used in uv mode):
        "s27":  Hallicrafters S-27 1 μV sensitivity reference line.
        "fubl2": FuBl 2 / EBL 3 3 mV operational reference line
                 (D.(Luft) T.4058 §21).
    """
    plt.style.use("dark_background")

    # Distance sweep
    d_min = 50
    d_km_fine = np.arange(d_min, d_max_km + 1, 10.0)

    # Friis / flat-Earth curve: link budget with zero diffraction loss.
    # This is the rectilinear propagation baseline — what the signal
    # WOULD be if there were no curvature and no residue-series decay.
    friis_snr = np.array([
        link_budget(d * 1000.0, 0.0, freq_mhz * 1e6, 0.0)["SNR_dB"]
        for d in d_km_fine
    ])

    # P.526 curve (SNR dB above receiver noise floor, globe).  Pass
    # the ground_name through so P.526 β / Eq. 18 G(Y) clamp match
    # the P.368 GRWAVE ground electrical parameters used below.
    p526_snr = np.array([
        p526_snr_peak(d, tx_m, rx_m, ground_name=ground_name,
                      freq_mhz=freq_mhz)
        for d in d_km_fine
    ])

    # Sommerfeld-Norton plane-earth curve. Rigorous flat-Earth
    # alternative to the single-ray Friis line. Includes direct +
    # ground-reflected + surface-wave terms, which produces Fresnel
    # lobes at short range and settles onto the Friis envelope past
    # ~300 km (see ITU Handbook on Ground Wave Propagation 2014
    # Part 1 §3.2.1, eq. 3-8).
    sn_snr = np.array([
        sn_snr_peak(d, tx_m, rx_m, ground_name=ground_name,
                    freq_mhz=freq_mhz)
        for d in d_km_fine
    ])

    # P.368 curve: grwave in one pass (SNR dB above receiver noise floor)
    d_grw, snr_grw = p368_sweep(tx_m, rx_m, ground_name, d_min, d_max_km,
                                 d_step_km=10.0, freq_mhz=freq_mhz)

    # Convert peak SNR arrays to EQUISIGNAL SNR by applying the 5° squint
    # crossover loss (CROSSOVER_dB = -19.0 dB). The equisignal is what
    # the pilot actually rides on; it sits 19 dB below either sub-beam
    # peak for the Telefunken Large Knickebein geometry. All four curves
    # in this graph represent equisignal V_rx at the aircraft receiver.
    friis_snr = friis_snr + CROSSOVER_dB
    sn_snr    = sn_snr + CROSSOVER_dB
    p526_snr  = p526_snr + CROSSOVER_dB
    snr_grw   = snr_grw + CROSSOVER_dB

    # SNR → V_rx conversion at 50Ω input:
    #   V_rx [dBμV] = P_rx [dBW] + 137   (physics: P->V at 50 Ω)
    #   SNR [dB]    = P_rx [dBW] - N_FLOOR_dBW
    #   → V_rx [dBμV] = SNR + (N_FLOOR_dBW + 137)
    # Computed dynamically from the library's N_FLOOR_dBW so any
    # bandwidth change in botb_itu_analysis.py propagates here.
    SNR_TO_DBUV = N_FLOOR_dBW + 137.0

    if y_mode == "uv":
        friis_plot = 10 ** ((friis_snr + SNR_TO_DBUV) / 20.0)
        sn_plot = 10 ** ((sn_snr + SNR_TO_DBUV) / 20.0)
        p526_plot = 10 ** ((p526_snr + SNR_TO_DBUV) / 20.0)
        snr_grw_plot = 10 ** ((snr_grw + SNR_TO_DBUV) / 20.0)
        noise_floor_y = 10 ** ((0.0 + SNR_TO_DBUV) / 20.0)   # ≈ 0.195 μV

        if receiver == "fubl2":
            # One extra reference line in fubl2 mode: Dan's 2 μV estimate
            # (from his link budget tool input on the 13 Feb 2025 DDS
            # stream at 2:37:10). Plotted alongside the noise floor so
            # the reader can see that even at Dan's own chosen sensitivity
            # level the globe case still fails at Stollberg → Midlands.
            rx_ref_y = 2.0
            rx_ref_label = "FuBl 2 (2 μV, Dan's estimate)"
            y_lim = (1e-4, 1e5)
        else:
            # Any other receiver mode (including the legacy "s27" and the
            # current "primary") renders with the physics noise floor as
            # the only reference line. We intentionally do not draw any
            # receiver-specific sensitivity line because no primary source
            # we have located quotes a bench microvolt MDS for any of the
            # candidate receivers (FuBl 1, FuBl 2, Hallicrafters S-27).
            rx_ref_y = None
            rx_ref_label = None
            y_lim = (1e-4, 1e5)

        y_label = "Equisignal voltage at receiver input (μV into 50 Ω, 5° squint)"
        use_log = True
    else:
        friis_plot = friis_snr
        sn_plot = sn_snr
        p526_plot = p526_snr
        snr_grw_plot = snr_grw
        noise_floor_y = 0.0
        rx_ref_y = None
        rx_ref_label = None
        y_label = "Equisignal SNR above receiver noise floor (dB, 5° squint)"
        y_lim = (-150.0, 150.0)
        use_log = False

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    if use_log:
        ax.set_yscale("log")

    ax.plot(d_km_fine, friis_plot, color=COL_FRIIS, linewidth=2.5,
            linestyle="-",
            label="Friis flat-Earth",
            zorder=4)
    ax.plot(d_km_fine, sn_plot, color=COL_SN, linewidth=2.5,
            linestyle="-",
            label="Sommerfeld-Norton FE",
            zorder=5)
    ax.plot(d_km_fine, p526_plot, color=COL_P526, linewidth=2.5,
            label="P.526 Fock (globe)", zorder=4)
    ax.plot(d_grw, snr_grw_plot, color=COL_P368, linewidth=2.5, linestyle="-",
            label="P.368 GRWAVE (globe)", zorder=4)

    # Noise + receiver reference lines (truncated labels)
    if y_mode == "uv":
        ax.axhline(y=noise_floor_y, color=COL_NOISE, linewidth=1.2,
                   linestyle="-", zorder=3,
                   label=f"Noise floor ({noise_floor_y:.2f} μV)")
        if rx_ref_y is not None:
            ref_color = COL_FUBL2 if receiver == "fubl2" else COL_DETECT
            # Abbreviate the receiver reference label for the legend
            if receiver == "fubl2":
                short_rx_ref_label = "FuBl 2 (2 μV, Dan)"
            else:
                short_rx_ref_label = "S-27 (1 μV, UK)"
            ax.axhline(y=rx_ref_y, color=ref_color, linewidth=1.4,
                       linestyle="--", alpha=0.85, zorder=3,
                       label=short_rx_ref_label)
    else:
        ax.axhline(y=0.0, color=COL_NOISE, linewidth=1.2, linestyle="-",
                   label="Noise floor (0 dB)", zorder=3)
        ax.axhline(y=DETECT_dB, color=COL_DETECT, linewidth=1.4,
                   linestyle="--", alpha=0.75,
                   label=f"Detection floor (+{DETECT_dB:.0f} dB)",
                   zorder=3)

    # Target markers. Give the first axvline a legend label so the dotted
    # target-distance line gets a single entry in the legend.
    #
    # The "t_y" field in each target tuple is the desired BOTTOM of the
    # rotated text (where the label visually starts, reading bottom to
    # top). Because rotation_mode='anchor' + va='bottom' places the
    # matplotlib anchor at the CENTER of the rotated bbox in y, we have
    # to compensate per-label: anchor_y = desired_bottom_y + half_width,
    # where half_width depends on the character count of each label. This
    # gives all labels the same visual starting y regardless of length.
    #
    # An optional 5th element in the target tuple is an x_offset
    # multiplier. Default 1.0. Labels that sit close to another label
    # (like Derby next to Beeston on the Stollberg graph) can use 0.5 to
    # halve the leftward offset and sit tighter to their own dotted line.
    x_span = d_max_km - d_min
    x_offset = 0.006 * x_span    # ~0.6% of x range, shifts text left of line

    # Empirical constant from a direct bbox measurement of matplotlib 8-pt
    # sans-serif text rotated 90° in a 6-inch tall figure with a 300 dB
    # y-axis range. Each character adds ~2.05 y-units to the rotated text
    # half-width.
    Y_PER_CHAR = 2.05

    # In uv (log-scale) mode we place target labels using a blended
    # transform so y is in axes fraction [0, 1] rather than data units.
    # The log scale breaks the per-label half-width compensation used
    # in snr mode, so uv mode uses a simpler fixed-position scheme.
    #
    # Data-coordinate target label placement for fubl2 sweep plots.
    # Each station has a default midband_y, but individual targets can
    # override via a 6th tuple element (t_custom_y) which is consulted
    # in the loop below. The defaults are:
    #   Kleve:     y = 0.02 μV, in the below-noise shaded region
    #              (1e-4 to 0.19 μV).
    #   Stollberg: y = 30 μV, in the clear band between the 2 μV
    #              FuBl 2 line and the 3000 μV Op Threshold.
    #   Telefunken: default value unused because every Telefunken
    #              target sets t_custom_y explicitly — the 400/500
    #              km targets go to 0.02 μV (noise floor band, same
    #              as Kleve) and the 700/800/1000 km targets go to
    #              30 μV (above-noise band, same as Stollberg).
    is_telefunken = "Telefunken" in station_name
    is_kleve = "Kleve" in station_name
    is_stollberg = "Stollberg" in station_name
    use_midband_labels = (y_mode == "uv" and receiver in ("fubl2", "primary"))
    if use_midband_labels and is_kleve:
        midband_y = 0.02
    else:
        midband_y = 30.0

    if y_mode == "uv":
        trans_labels = blended_transform_factory(ax.transData, ax.transAxes)
        UV_LABEL_Y_BOTTOM = 0.02
        UV_LABEL_Y_TOP = 0.65

    for i, target in enumerate(targets):
        t_xoff_mult = 1.0
        t_custom_y = None    # per-target midband y override (data coords)
        if len(target) == 6:
            t_name, t_d, t_label, t_y, t_xoff_mult, t_custom_y = target
        elif len(target) == 5:
            t_name, t_d, t_label, t_y, t_xoff_mult = target
        elif len(target) == 4:
            t_name, t_d, t_label, t_y = target
        else:
            t_name, t_d, t_label = target
            t_y = -110   # default

        # Only include the target name if it adds information beyond the
        # distance itself. For the Telefunken series the target "name" is
        # the same number as the distance, so we skip it to avoid "400 400
        # km" redundancy.
        if t_name and str(t_name).strip() != str(t_d):
            text = f"{t_name} {t_d} km"
        else:
            text = f"{t_d} km"

        lbl = "Target" if i == 0 else None
        ax.axvline(x=t_d, color=COL_TARGET, linewidth=1.3, linestyle=":",
                   alpha=0.7, zorder=2, label=lbl)

        if use_midband_labels:
            # Data-coordinate y placement for FuBl 2 sweep plots.
            # Per-target custom y (6th tuple element) overrides the
            # station default midband_y.
            y_pos = t_custom_y if t_custom_y is not None else midband_y
            ax.text(t_d - x_offset * t_xoff_mult, y_pos, text,
                    rotation=90, rotation_mode="anchor",
                    ha="center", va="bottom", fontsize=8,
                    color="white", alpha=1.0, zorder=10,
                    fontweight="bold")
        elif y_mode == "uv":
            # Data-coordinate y placement for "below" labels (t_y < 0
            # in snr semantics). The right anchor depends on the
            # station, because each station's curves clear different
            # parts of the y axis:
            #   Kleve (overland): both globe curves stay well above
            #     the noise floor at every Midlands target, so the
            #     labels go INSIDE the shaded below-noise band at
            #     y ≈ 5e-3 μV (below the curves, no overlap).
            #   Stollberg (oversea): both globe curves have plunged
            #     past the noise floor by the time they reach the
            #     Midlands targets, so there's a clear band between
            #     ~0.2 μV and ~1000 μV where no curve passes. Anchor
            #     labels at y = 1.0 μV in that empty band.
            #   Other (Telefunken, etc.): default to the Kleve-style
            #     placement at y = 5e-3 μV.
            # "Above" labels (t_y >= 0) keep the axes-fraction top
            # placement.
            if t_y < 0:
                if is_stollberg:
                    # Stollberg: empty band between the high-signal
                    # Friis/SN curves and the dropped globe curves.
                    # y = 30 μV sits comfortably between the noise
                    # floor (0.19 μV) and the upper curves
                    # (~5000-10000 μV).
                    y_data = 30.0
                elif is_telefunken:
                    # Telefunken 400/500 km only (700/800/1000 km use
                    # the "above" branch via t_y >= 0). Anchor in the
                    # upper part of the shaded below-noise band so the
                    # labels sit between the legend block and the
                    # noise floor line.
                    y_data = 1e-2
                else:
                    # Kleve and other land paths: labels go INSIDE the
                    # shaded below-noise band, above the legend block.
                    y_data = 5e-3
                ax.text(t_d - x_offset * t_xoff_mult, y_data, text,
                        rotation=90, rotation_mode="anchor",
                        ha="center", va="bottom", fontsize=8,
                        color="white", alpha=1.0, zorder=10,
                        fontweight="bold")
            else:
                ax.text(t_d - x_offset * t_xoff_mult, UV_LABEL_Y_TOP, text,
                        rotation=90, rotation_mode="anchor",
                        ha="center", va="bottom", fontsize=8,
                        color="white", alpha=1.0, zorder=10,
                        fontweight="bold",
                        transform=trans_labels)
        else:
            # Linear dB axis. Compute per-label anchor y so the visible
            # bottom of the rotated text lands at exactly t_y regardless
            # of character count.
            half_width = len(text) * Y_PER_CHAR
            anchor_y = t_y + half_width
            ax.text(t_d - x_offset * t_xoff_mult, anchor_y, text,
                    rotation=90, rotation_mode="anchor",
                    ha="center", va="bottom", fontsize=8,
                    color="white", alpha=1.0, zorder=10,
                    fontweight="bold")

    # Radio horizon marker
    d_los = line_of_sight(tx_m, rx_m) / 1000.0
    ax.axvline(x=d_los, color="#888888", linewidth=1.2,
               linestyle=(0, (6, 4)), alpha=0.8, zorder=1,
               label=f"Horizon ({d_los:.0f} km)")

    # Shade below noise faintly. In uv mode "below noise" is below the
    # ~0.195 μV line; in snr mode it is below y=0 dB.
    if y_mode == "uv":
        ax.axhspan(y_lim[0], noise_floor_y, alpha=0.08,
                   color="#FFD54F", zorder=0)
    else:
        ax.axhspan(-300, 0.0, alpha=0.08, color="#FFD54F", zorder=0)

    ax.set_xlabel("Distance (km)", fontsize=11, color="white")
    ax.set_ylabel(y_label, fontsize=11, color="white")

    title_line1 = (f"{station_name}: Friis + Sommerfeld-Norton FE "
                   f"vs P.526 Fock vs P.368 GRWAVE")
    title_line2 = (f"TX {tx_m} m  |  RX {rx_m:,} m  |  "
                   f"{freq_mhz:.1f} MHz, 3 kW, 26 dBi  |  "
                   f"{GROUND[ground_name]['label']}{title_suffix}")
    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=13,
                 fontweight="bold", color="white", pad=10)

    ax.set_xlim(d_min, d_max_km)
    ax.set_ylim(*y_lim)
    ax.grid(alpha=0.2, color="gray", which="both")
    ax.tick_params(colors="white")

    # In uv mode, format log-axis ticks as human-readable μV values.
    if y_mode == "uv":
        def fmt_uv(val, pos):
            if val >= 1:
                return f"{val:g} μV"
            elif val >= 0.001:
                return f"{val:g} μV"
            else:
                return f"{val:.0e} μV"
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_uv))

    # Legend in the lower-left corner of the plot. Sits in the shadow
    # region below the signal curves where it does not overlap data.
    ax.legend(loc="lower left", fontsize=8, framealpha=0.55,
              facecolor=BG_COLOR, edgecolor="#444444", labelcolor="white",
              ncol=2)

    fig.tight_layout()
    save(fig, fname)


def plot_master_bargraph(fname, y_mode="snr", receiver="s27",
                         include_sn=True):
    """Side-by-side bar chart of flat-Earth + globe models for all
    confirmed Knickebein + Telefunken paths.

    include_sn: If True (default), the chart has four bars per path
    (Friis flat, Sommerfeld-Norton FE, P.526 Fock, P.368 GRWAVE). If
    False, the Sommerfeld-Norton column is omitted and the chart
    shows only Friis + P.526 + P.368 (the simpler "lead" variant).

    y_mode: "snr" (default) plots peak SNR above receiver noise floor in
    dB on a linear axis. "uv" plots peak voltage at the 50 Ω receiver
    input in μV on a log axis.
    receiver: "s27" (default) shows a 1 μV S-27 sensitivity line in uv
    mode. "fubl2" shows a 3 mV FuBl 2 / EBL 3 operational reference line.
    """
    plt.style.use("dark_background")

    # Concatenate Kleve, Stollberg, and Telefunken paths in a single order
    all_paths = KN_PATHS + TF_PATHS

    labels = []
    friis_snr = []
    sn_snr = []
    p526_snr = []
    p368_snr = []

    for name, d_km, tx, rx, note, pt in all_paths:
        labels.append(name)
        # Friis flat-Earth: pure FSPL link budget (no diffraction loss).
        friis_snr.append(
            link_budget(d_km * 1000.0, 0.0, FREQ_MHz * 1e6, 0.0)["SNR_dB"]
        )
        # Sommerfeld-Norton plane Earth: rigorous flat-Earth solution
        # (direct + reflected + surface wave).
        sn_snr.append(sn_snr_peak(d_km, tx, rx, ground_name=pt))
        # P.526 must use the same ground type as P.368 so that β / Eq. 18
        # G(Y) clamp tracks the electrical ground model (matters for
        # 31.5 MHz vert pol over sea: below the 300 MHz cut).
        p526_snr.append(p526_snr_peak(d_km, tx, rx, ground_name=pt))
        p368_snr.append(p368_snr_peak(d_km, tx, rx, pt))

    friis_snr = np.array(friis_snr)
    sn_snr = np.array(sn_snr)
    p526_snr = np.array(p526_snr)
    p368_snr = np.array(p368_snr)

    # Apply equisignal crossover loss (5° squint, -19 dB) to every bar.
    # The bars then represent equisignal V_rx at the aircraft receiver,
    # matching the per-station sweep graphs and the null-doc table.
    friis_snr = friis_snr + CROSSOVER_dB
    sn_snr    = sn_snr + CROSSOVER_dB
    p526_snr  = p526_snr + CROSSOVER_dB
    p368_snr  = p368_snr + CROSSOVER_dB

    if y_mode == "uv":
        # SNR → V_rx(μV) at 50 Ω: V_rx_dBμV = SNR + (N_FLOOR_dBW + 137).
        # Computed dynamically from the library so bandwidth changes
        # propagate here.
        SNR_TO_DBUV = N_FLOOR_dBW + 137.0
        friis_vals = 10 ** ((friis_snr + SNR_TO_DBUV) / 20.0)
        sn_vals = 10 ** ((sn_snr + SNR_TO_DBUV) / 20.0)
        p526_vals = 10 ** ((p526_snr + SNR_TO_DBUV) / 20.0)
        p368_vals = 10 ** ((p368_snr + SNR_TO_DBUV) / 20.0)
        noise_floor_y = 10 ** ((0.0 + SNR_TO_DBUV) / 20.0)

        if receiver == "fubl2":
            rx_ref_y = 2.0
            rx_ref_label = "FuBl 2 (2 μV, Dan's estimate)"
            ref_color = COL_FUBL2
        else:
            # Any non-fubl2 mode (including the legacy "s27" and the
            # current "primary") renders with the physics noise floor as
            # the only reference line. We intentionally do not draw any
            # receiver-specific sensitivity line because no primary source
            # we have located quotes a bench microvolt MDS for any of the
            # candidate receivers (FuBl 1, FuBl 2, Hallicrafters S-27).
            rx_ref_y = None
            rx_ref_label = None
            ref_color = None

        y_lim = (1e-6, 1e6)
        y_label = "Equisignal voltage at receiver input (μV into 50 Ω, 5° squint)"
        use_log = True
        floor_label_noise = f"Noise floor ({noise_floor_y:.2f} μV @ 50 Ω)"
        title_axis_note = ("equisignal voltage at the receiver input in μV "
                            "(50 Ω, 5° squint, log scale)")
    else:
        friis_vals = friis_snr
        sn_vals = sn_snr
        p526_vals = p526_snr
        p368_vals = p368_snr
        noise_floor_y = 0.0
        # Detection-floor reference line intentionally disabled in dB mode
        # per user request. The noise floor itself at 0 dB is the only
        # horizontal reference the viewer needs.
        rx_ref_y = None
        rx_ref_label = None
        y_label = "Equisignal SNR above receiver noise floor (dB, 5° squint)"
        y_lim = None
        use_log = False
        floor_label_noise = "Noise floor"
        ref_color = None
        title_axis_note = "equisignal SNR above receiver noise floor in dB"

    fig, ax = plt.subplots(figsize=(18, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    if use_log:
        ax.set_yscale("log")

    x = np.arange(len(labels))
    # 4 bars per path when include_sn, 3 when not
    w = 0.20 if include_sn else 0.27

    # On a log bar chart matplotlib treats bar values as the top of each
    # bar with the baseline at bottom=1 by default, which makes bars
    # below 1 μV point downward. Anchor the baseline at the bottom of the
    # y range in uv mode so bars rise from a consistent floor.
    bar_bottom = y_lim[0] if use_log else 0.0

    if include_sn:
        offsets = (-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w)
    else:
        offsets = (-w, 0.0, w, None)   # no SN column

    bars0 = ax.bar(x + offsets[0], friis_vals, w,
                   bottom=bar_bottom if use_log else 0,
                   color=COL_FRIIS, alpha=0.9,
                   label="Friis flat-Earth (rectilinear)",
                   edgecolor="white", linewidth=0.5, zorder=3)
    bars_sn = None
    if include_sn:
        bars_sn = ax.bar(x + offsets[1], sn_vals, w,
                       bottom=bar_bottom if use_log else 0,
                       color=COL_SN, alpha=0.9,
                       label="Sommerfeld-Norton FE",
                       edgecolor="white", linewidth=0.5, zorder=3)
    p526_off = offsets[2] if include_sn else offsets[1]
    p368_off = offsets[3] if include_sn else offsets[2]
    bars1 = ax.bar(x + p526_off, p526_vals, w,
                   bottom=bar_bottom if use_log else 0,
                   color=COL_P526, alpha=0.9,
                   label="ITU-R P.526-16 (Fock diffraction, globe)",
                   edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + p368_off, p368_vals, w,
                   bottom=bar_bottom if use_log else 0,
                   color=COL_P368, alpha=0.9,
                   label="ITU-R P.368 GRWAVE (ground wave, globe)",
                   edgecolor="white", linewidth=0.5, zorder=3)

    # Noise + receiver reference lines. Keep references to each line
    # artist so we can assemble the legend in the exact order we want
    # below (column 1 = reference lines, column 2 = bar colours).
    noise_line = ax.axhline(y=noise_floor_y, color=COL_NOISE, linewidth=1.2,
                            linestyle="-", label=floor_label_noise, zorder=2)
    rx_line = None
    if rx_ref_y is not None:
        rx_line = ax.axhline(y=rx_ref_y, color=ref_color, linewidth=1.4,
                             linestyle="--", alpha=0.85, zorder=2,
                             label=rx_ref_label)

    # Value labels on each bar
    bar_value_pairs = [(bars0, friis_vals)]
    if bars_sn is not None:
        bar_value_pairs.append((bars_sn, sn_vals))
    bar_value_pairs.extend([(bars1, p526_vals), (bars2, p368_vals)])
    for bars, vals in bar_value_pairs:
        for bar, val in zip(bars, vals):
            if use_log:
                y_pos = val * 1.2
                if y_pos < y_lim[0]:
                    y_pos = y_lim[0] * 1.5
                if val >= 10:
                    txt = f"{val:.0f}"
                elif val >= 1:
                    txt = f"{val:.1f}"
                else:
                    txt = f"{val:.1e}"
                ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                        txt, ha="center", va="bottom", fontsize=6,
                        fontweight="bold", color="white", rotation=90)
            else:
                y_off = 3 if val >= noise_floor_y else -3
                va = "bottom" if val >= noise_floor_y else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                        f"{val:+.0f}", ha="center", va=va, fontsize=7,
                        fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9,
                       color="white")
    ax.set_ylabel(y_label, fontsize=12, color="white")

    if y_lim:
        ax.set_ylim(*y_lim)

    fe_label = "Friis + Sommerfeld-Norton FE" if include_sn else "Friis flat-Earth"
    if y_mode == "uv":
        title_short = (
            f"Battle of the Beams: {fe_label} vs "
            "ITU-R P.526 Fock vs P.368 GRWAVE\n"
            "31.5 MHz, 3 kW, 26 dBi TX — equisignal V_rx at 50 Ω (5° squint, log scale)"
        )
    else:
        title_short = (
            f"Battle of the Beams: {fe_label} vs "
            "ITU-R P.526 Fock vs P.368 GRWAVE\n"
            "31.5 MHz, 3 kW, 26 dBi TX — equisignal SNR above receiver noise (5° squint)"
        )
    ax.set_title(title_short, fontsize=13, fontweight="bold",
                 color="white", pad=20)
    ax.grid(axis="y", alpha=0.2, color="gray", zorder=0, which="both")
    ax.tick_params(colors="white")

    # Format y-axis ticks as μV labels in uv mode.
    if y_mode == "uv":
        def fmt_uv(val, pos):
            if val >= 1:
                return f"{val:g} μV"
            elif val >= 0.001:
                return f"{val:g} μV"
            else:
                return f"{val:.0e} μV"
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_uv))

    # Legend INSIDE the plot at upper-right, ncol=2 (column-major fill).
    # FE models on the left column, GE models on the right column, so the
    # reader reads flat-Earth on one side and globe on the other.
    # When include_sn=True (4-color), layout is:
    #     Noise floor              (blank)
    #     Sommerfeld-Norton FE     ITU-R P.526-16
    #     Friis flat-Earth         ITU-R P.368
    # When include_sn=False (3-color), layout is 2x2:
    #     Noise floor              ITU-R P.526-16
    #     Friis flat-Earth         ITU-R P.368
    blank = Patch(visible=False)
    if bars_sn is not None:
        # Column 1 (left, FE side): noise floor, SN, Friis
        col1_handles = [noise_line, bars_sn, bars0]
        col1_labels  = [floor_label_noise, bars_sn.get_label(), bars0.get_label()]
        # Column 2 (right, GE side): blank, P.526, P.368
        col2_handles = [blank, bars1, bars2]
        col2_labels  = ["", bars1.get_label(), bars2.get_label()]
    else:
        # Column 1 (left, FE side): noise floor, Friis
        col1_handles = [noise_line, bars0]
        col1_labels  = [floor_label_noise, bars0.get_label()]
        # Column 2 (right, GE side): P.526, P.368
        col2_handles = [bars1, bars2]
        col2_labels  = [bars1.get_label(), bars2.get_label()]
    legend_handles = col1_handles + col2_handles
    legend_labels  = col1_labels + col2_labels
    ax.legend(legend_handles, legend_labels,
              loc="upper right", fontsize=10, framealpha=0.55,
              facecolor=BG_COLOR, edgecolor="#444444", labelcolor="white",
              ncol=2)

    fig.tight_layout()
    save(fig, fname)


# ================================================================
#  MAIN
# ================================================================

def main():
    print("Generating P.526 vs P.368 comparison graphs...")

    # Kleve sweep. Five Kleve-specific target/measurement points:
    #   Spalding (440 km): where Bufton's 21 Jun 1940 Anson flight
    #       measured the Kleve director beam's 400–500 yd equisignal
    #       width.
    #   Retford (512 km): Enigma decrypts named Retford as a Kleve
    #       beam pointing coordinate.
    #   Derby (530 km): operational target (Rolls-Royce Merlin works).
    #       This is where Kleve was actually aimed — the director beam
    #       for every Midlands raid. Stollberg's cross beam crossed
    #       Kleve over Derby, not over Beeston.
    #   Birmingham (551 km): operational target.
    #   Liverpool (640 km): operational target.
    # Beeston is NOT a Kleve target. It is on the Stollberg graph as a
    # measurement site: Bufton detected the Stollberg cross beam at
    # Beeston while flying north along the Kleve director beam.
    kleve_targets = [
        ("Spalding",   440, "", -140),
        ("Retford",    512, "", -140),
        ("Derby",      530, "", -140, 0.5),
        ("Birmingham", 551, "", -140),
        ("Liverpool",  640, "", -140),
    ]
    print("\n[1/13] Kleve sweep (SNR axis)")
    plot_station_sweep("Kleve → Midlands", 111, 6000, "land",
                       FREQ_MHz, 900, kleve_targets,
                       "p526_vs_p368_kleve.png")

    # Stollberg sweep. Four Stollberg-specific target/measurement
    # points:
    #   Beeston (694 km): measurement site. Bufton's 21 Jun 1940 Anson
    #       flight detected the Stollberg cross beam at Beeston while
    #       flying north along the Kleve director beam. Beeston is not
    #       an operational target.
    #   Derby (710 km): operational target. Stollberg's cross beam
    #       was aimed to intersect Kleve's director beam over Derby.
    #   Birmingham (754 km): operational target.
    #   Liverpool (791 km): operational target.
    # Spalding and Retford are NOT on the Stollberg graph — Spalding is
    # where the Kleve beam was measured, Retford is an Enigma intercept
    # for the Kleve beam; neither has any operational link to Stollberg.
    # Derby gets a halved x_offset (0.5×) because it sits only 16 km
    # from Beeston on the x axis.
    stoll_targets = [
        ("Beeston",    694, "", -140),
        ("Derby",      710, "", -140, 0.5),
        ("Birmingham", 754, "", -140),
        ("Liverpool",  791, "", -140),
    ]
    print("\n[2/13] Stollberg sweep (SNR axis)")
    plot_station_sweep("Stollberg → Midlands", 72, 6000, "sea",
                       FREQ_MHz, 1000, stoll_targets,
                       "p526_vs_p368_stollberg.png")

    # Telefunken over-sea sweep. Per-target data-coordinate y placement
    # for fubl2 mode (6th tuple element = midband y in μV):
    #   400/500 km → y = 0.02 μV, inside the below-noise shaded region
    #                (text spans ~0.02 to ~0.14 μV on a log axis).
    #   700/800/1000 km → y = 30 μV, in the clear band between the
    #                     2 μV FuBl 2 line and the 3000 μV Op Threshold.
    # The t_y "snr anchor" value is retained for snr-axis compatibility.
    tf_targets = [
        ("", 400,  "", -75, 1.0, 0.02),
        ("", 500,  "", -75, 1.0, 0.02),
        ("", 700,  "",  30, 1.0, 30.0),
        ("", 800,  "",  30, 1.0, 30.0),
        ("", 1000, "",  30, 1.0, 30.0),
    ]
    print("\n[3/13] Telefunken sweep (SNR axis)")
    plot_station_sweep("Telefunken July 1939 over-sea tests", 72, 4000,
                       "sea", FREQ_MHz, 1200, tf_targets,
                       "p526_vs_p368_telefunken.png",
                       title_suffix="  (BArch RL 19-6/40 ref. 230Q8 App. 2)")

    # Master bar chart
    print("\n[4/13] Master bar chart (SNR axis)")
    plot_master_bargraph("p526_vs_p368_master_bargraph.png")

    # ------------------------------------------------------------
    #  Microvolt / S-27 versions (British receiver reference).
    #  Y axis in linear μV at the 50 Ω receiver input, log scale.
    #  Reference line is the Hallicrafters S-27 sensitivity (1 μV).
    # ------------------------------------------------------------
    print("\n[5/13] Kleve sweep (μV axis, S-27)")
    plot_station_sweep("Kleve → Midlands", 111, 6000, "land",
                       FREQ_MHz, 900, kleve_targets,
                       "p526_vs_p368_uv_kleve.png",
                       y_mode="uv", receiver="s27")

    print("\n[6/13] Stollberg sweep (μV axis, S-27)")
    plot_station_sweep("Stollberg → Midlands", 72, 6000, "sea",
                       FREQ_MHz, 1000, stoll_targets,
                       "p526_vs_p368_uv_stollberg.png",
                       y_mode="uv", receiver="s27")

    print("\n[7/13] Telefunken sweep (μV axis, S-27)")
    plot_station_sweep("Telefunken July 1939 over-sea tests", 72, 4000,
                       "sea", FREQ_MHz, 1200, tf_targets,
                       "p526_vs_p368_uv_telefunken.png",
                       title_suffix="  (BArch RL 19-6/40 ref. 230Q8 App. 2)",
                       y_mode="uv", receiver="s27")

    print("\n[7a] Master bar chart (μV axis, Friis-only lead variant)")
    plot_master_bargraph("p526_vs_p368_uv_master_bargraph_friis.png",
                         y_mode="uv", receiver="s27", include_sn=False)

    print("\n[8/13] Master bar chart (μV axis, with Sommerfeld-Norton FE)")
    plot_master_bargraph("p526_vs_p368_uv_master_bargraph.png",
                         y_mode="uv", receiver="s27")

    # ------------------------------------------------------------
    #  Microvolt / primary-source reference versions.
    #  Same y axis, same calculation, but the receiver reference set
    #  is restricted to primary-source-derivable lines only: the
    #  thermal+galactic noise floor and the Luftwaffe §21 operational
    #  guarantee at 3 mV (derived from D.(Luft) T.4058 §21 with 500 W
    #  Lorenz / 70 km / 200 m alt). No Dan 2 μV "FuBl 2 estimate"
    #  line — that one is retained only on the separate fubl2 master
    #  bar chart as a Dan comparison.
    # ------------------------------------------------------------
    print("\n[9/13] Kleve sweep (μV axis, primary)")
    plot_station_sweep("Kleve → Midlands", 111, 6000, "land",
                       FREQ_MHz, 900, kleve_targets,
                       "p526_vs_p368_fubl2_kleve.png",
                       y_mode="uv", receiver="primary")

    print("\n[10/13] Stollberg sweep (μV axis, primary)")
    plot_station_sweep("Stollberg → Midlands", 72, 6000, "sea",
                       FREQ_MHz, 1000, stoll_targets,
                       "p526_vs_p368_fubl2_stollberg.png",
                       y_mode="uv", receiver="primary")

    print("\n[11/13] Telefunken sweep (μV axis, primary)")
    plot_station_sweep("Telefunken July 1939 over-sea tests", 72, 4000,
                       "sea", FREQ_MHz, 1200, tf_targets,
                       "p526_vs_p368_fubl2_telefunken.png",
                       title_suffix="  (BArch RL 19-6/40 ref. 230Q8 App. 2)",
                       y_mode="uv", receiver="primary")

    print("\n[12/13] Primary master bar chart (μV axis, no Dan line)")
    plot_master_bargraph("p526_vs_p368_primary_master_bargraph.png",
                         y_mode="uv", receiver="primary")

    print("\n[13/13] FuBl 2 master bar chart (μV axis, Dan comparison)")
    plot_master_bargraph("p526_vs_p368_fubl2_master_bargraph.png",
                         y_mode="uv", receiver="fubl2")

    print("\nDone.")


if __name__ == "__main__":
    main()
