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

# Import grwave FIRST, before touching sys.path. The /home/alan/claude/BotB/
# directory contains a local grwave/ subfolder (the cloned repo) which would
# shadow the installed grwave package as a namespace package if BotB is on
# sys.path. So we import grwave cleanly first, then add BotB to sys.path for
# botb_itu_analysis.
import grwave as grwave_pkg

sys.path.insert(0, "/home/alan/claude/BotB")

from botb_itu_analysis import (
    itu_diffraction_loss,
    link_budget,
    line_of_sight,
    CROSSOVER_dB,
    N_FLOOR_dBW,
    G_DIR_dB,
    DETECT_dB,
)


BOTB_DIR = "/home/alan/claude/BotB"
VAULT_DIR = "/home/alan/Documents/multi_2/Attachments"

# Common constants
FREQ_MHz = 31.5
TX_POWER_W = 3000          # Large Knickebein Telefunken spec

# Style
BG_COLOR = "#1a1a1a"
COL_P526 = "#FF1493"       # magenta (Fock diffraction, existing style)
COL_P368 = "#00E5FF"       # cyan (new: ITU P.368 ground-wave)
COL_NOISE = "white"
COL_DETECT = "#FFEB3B"     # faint yellow (detection floor)
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

def p526_snr_peak(d_km, tx_m, rx_m, freq_mhz=FREQ_MHz, rx_gain=0.0):
    """ITU-R P.526 globe-model peak SNR (no equisignal crossover subtracted)
    at a single distance, using the same parameters as botb_itu_analysis."""
    d_m = d_km * 1000.0
    freq_hz = freq_mhz * 1e6
    itu = itu_diffraction_loss(d_m, tx_m, rx_m, freq_hz)
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
                       d_max_km, targets, fname, title_suffix=""):
    """Distance sweep graph comparing P.526 peak SNR and P.368 peak SNR
    for a single transmitter, with target distance markers."""
    plt.style.use("dark_background")

    # Distance sweep
    d_min = 50
    d_km_fine = np.arange(d_min, d_max_km + 1, 10.0)

    # P.526 curve
    p526_snr = np.array([p526_snr_peak(d, tx_m, rx_m, freq_mhz) for d in d_km_fine])

    # P.368 curve: grwave in one pass
    d_grw, snr_grw = p368_sweep(tx_m, rx_m, ground_name, d_min, d_max_km,
                                 d_step_km=10.0, freq_mhz=freq_mhz)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(d_km_fine, p526_snr, color=COL_P526, linewidth=2.5,
            label="ITU-R P.526-16 (Fock diffraction)", zorder=4)
    ax.plot(d_grw, snr_grw, color=COL_P368, linewidth=2.5, linestyle="-",
            label=f"ITU-R P.368 GRWAVE ({GROUND[ground_name]['label']})",
            zorder=4)

    # Noise + detection floor
    ax.axhline(y=0, color=COL_NOISE, linewidth=1.2, linestyle="-",
               label="Noise floor (0 dB)", zorder=3)
    ax.axhline(y=DETECT_dB, color=COL_DETECT, linewidth=1.4, linestyle="--",
               alpha=0.75,
               label=f"Detection floor ({DETECT_dB:.0f} dB above noise)",
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

    for i, target in enumerate(targets):
        t_xoff_mult = 1.0
        if len(target) == 5:
            t_name, t_d, t_label, t_y, t_xoff_mult = target
        elif len(target) == 4:
            t_name, t_d, t_label, t_y = target
        else:
            t_name, t_d, t_label = target
            t_y = -110   # default: start at -110 dB, extend upward

        # Only include the target name if it adds information beyond the
        # distance itself. For the Telefunken series the target "name" is
        # the same number as the distance, so we skip it to avoid "400 400
        # km" redundancy.
        if t_name and str(t_name).strip() != str(t_d):
            text = f"{t_name} {t_d} km"
        else:
            text = f"{t_d} km"

        # Compute per-label anchor y so the visible bottom of the rotated
        # text lands at exactly t_y regardless of character count.
        half_width = len(text) * Y_PER_CHAR
        anchor_y = t_y + half_width

        lbl = "Target distance" if i == 0 else None
        ax.axvline(x=t_d, color=COL_TARGET, linewidth=1.3, linestyle=":",
                   alpha=0.7, zorder=2, label=lbl)
        # Target-name text is painted bright white at zorder=10 so it
        # sits above every other artist including the faint yellow shaded
        # below-noise region (zorder=0), the signal curves, the horizon
        # markers, and the detection-floor line. Pure white plus explicit
        # alpha=1.0 keeps the labels high-contrast against the dark bg.
        ax.text(t_d - x_offset * t_xoff_mult, anchor_y, text,
                rotation=90, rotation_mode="anchor",
                ha="center", va="bottom", fontsize=8,
                color="white", alpha=1.0, zorder=10,
                fontweight="bold")

    # Radio horizon marker
    d_los = line_of_sight(tx_m, rx_m) / 1000.0
    ax.axvline(x=d_los, color="#888888", linewidth=1.2,
               linestyle=(0, (6, 4)), alpha=0.8, zorder=1,
               label=f"Radio horizon: {d_los:.0f} km")

    # Shade below noise faintly
    ax.axhspan(-300, 0, alpha=0.08, color="#FFD54F", zorder=0)

    ax.set_xlabel("Distance (km)", fontsize=11, color="white")
    ax.set_ylabel("Peak SNR above receiver noise floor (dB)",
                  fontsize=11, color="white")

    title_line1 = f"{station_name}: ITU P.526 vs ITU P.368 GRWAVE"
    title_line2 = (f"TX {tx_m} m  |  RX {rx_m:,} m  |  "
                   f"{freq_mhz:.1f} MHz  |  TX 3 kW, 26 dBi directional  |  "
                   f"{GROUND[ground_name]['label']}{title_suffix}")
    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=13,
                 fontweight="bold", color="white", pad=10)

    ax.set_xlim(d_min, d_max_km)
    ax.set_ylim(-150, 150)
    ax.grid(alpha=0.2, color="gray")
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.35,
              facecolor=BG_COLOR, edgecolor="#444444", labelcolor="white")

    fig.tight_layout()
    save(fig, fname)


def plot_master_bargraph(fname):
    """Side-by-side bar chart of P.526 peak SNR vs P.368 peak SNR for all
    confirmed Knickebein + Telefunken paths."""
    plt.style.use("dark_background")

    # Concatenate Kleve, Stollberg, and Telefunken paths in a single order
    all_paths = KN_PATHS + TF_PATHS

    labels = []
    p526_vals = []
    p368_vals = []

    for name, d_km, tx, rx, note, pt in all_paths:
        labels.append(name)
        p526_vals.append(p526_snr_peak(d_km, tx, rx))
        p368_vals.append(p368_snr_peak(d_km, tx, rx, pt))

    p526_vals = np.array(p526_vals)
    p368_vals = np.array(p368_vals)

    fig, ax = plt.subplots(figsize=(16, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x = np.arange(len(labels))
    w = 0.4

    bars1 = ax.bar(x - w/2, p526_vals, w, color=COL_P526, alpha=0.9,
                   label="ITU-R P.526-16 (Fock diffraction)",
                   edgecolor="white", linewidth=0.5, zorder=3)
    bars2 = ax.bar(x + w/2, p368_vals, w, color=COL_P368, alpha=0.9,
                   label="ITU-R P.368 GRWAVE (ground wave)",
                   edgecolor="white", linewidth=0.5, zorder=3)

    # Noise + detection floor
    ax.axhline(y=0, color=COL_NOISE, linewidth=1.2, linestyle="-",
               label="Noise floor (0 dB)", zorder=2)
    ax.axhline(y=DETECT_dB, color=COL_DETECT, linewidth=1.4, linestyle="--",
               alpha=0.75, zorder=2,
               label=f"Detection floor (+{DETECT_dB:.0f} dB)")

    # Value labels
    for bars, vals in [(bars1, p526_vals), (bars2, p368_vals)]:
        for bar, val in zip(bars, vals):
            y_off = 3 if val >= 0 else -3
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                    f"{val:+.0f}", ha="center", va=va, fontsize=8,
                    fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9,
                       color="white")
    ax.set_ylabel("Peak SNR above receiver noise floor (dB)",
                  fontsize=12, color="white")
    ax.set_title(
        "Battle of the Beams: ITU-R P.526 vs ITU-R P.368 "
        "at 31.5 MHz, 3 kW, 26 dBi directional TX\n"
        "Both are ITU international standards. P.526 is Fock smooth-Earth "
        "diffraction; P.368 GRWAVE is ground-wave propagation "
        "(ITU's own Fortran code).",
        fontsize=13, fontweight="bold", color="white", pad=12)
    ax.grid(axis="y", alpha=0.2, color="gray", zorder=0)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.35,
              facecolor=BG_COLOR, edgecolor="#444444", labelcolor="white")

    fig.tight_layout()
    save(fig, fname)


# ================================================================
#  MAIN
# ================================================================

def main():
    print("Generating P.526 vs P.368 comparison graphs...")

    # Kleve sweep. Same 6 Knickebein Midlands-complex targets as the
    # Stollberg graph, with one visual merge: Retford (511.9 km) and
    # Beeston (512.6 km) are 0.7 km apart from Kleve's perspective so
    # they share a single dotted line and a combined label. Derby is at
    # 0.5× offset because it sits 17 km from the Retford/Beeston merged
    # marker. All label bottoms at y=-140.
    kleve_targets = [
        ("Spalding",           440, "", -140),
        ("Retford/Beeston",    512, "", -140),
        ("Derby",              530, "", -140, 0.5),
        ("Birmingham",         551, "", -140),
        ("Liverpool",          640, "", -140),
    ]
    print("\n[1/4] Kleve sweep (111 m TX, 6000 m RX, overland)")
    plot_station_sweep("Kleve → Midlands", 111, 6000, "land",
                       FREQ_MHz, 900, kleve_targets,
                       "p526_vs_p368_kleve.png")

    # Stollberg sweep. Full set of 6 Knickebein Midlands-complex targets
    # matching the Kleve graph. Derby gets a halved x_offset (0.5×)
    # because it sits only 16 km from Beeston on the x axis. All label
    # bottoms at y=-140.
    stoll_targets = [
        ("Spalding",   633, "", -140),
        ("Retford",    664, "", -140),
        ("Beeston",    694, "", -140),
        ("Derby",      710, "", -140, 0.5),
        ("Birmingham", 754, "", -140),
        ("Liverpool",  791, "", -140),
    ]
    print("\n[2/4] Stollberg sweep (72 m TX, 6000 m RX, mixed sea)")
    plot_station_sweep("Stollberg → Midlands", 72, 6000, "sea",
                       FREQ_MHz, 1000, stoll_targets,
                       "p526_vs_p368_stollberg.png")

    # Telefunken over-sea sweep. 400/500 km labels in the bottom half.
    # 700/800/1000 km labels sit with their rotated text BOTTOM around
    # y=35 (25 dB above the +10 detection floor line).
    #
    # Empirically measured: with rotation=90, rotation_mode='anchor',
    # va='bottom', the bottom of the rotated bbox lands at ~(anchor - 15)
    # for these 6-7 character labels. So anchor=+50 puts the bottoms at
    # ~y=35 clear above the detection floor.
    tf_targets = [
        ("", 400,  "", -75),
        ("", 500,  "", -75),
        ("", 700,  "",  30),
        ("", 800,  "",  30),
        ("", 1000, "",  30),
    ]
    print("\n[3/4] Telefunken sweep (72 m TX, 4000 m RX, open sea)")
    plot_station_sweep("Telefunken July 1939 over-sea tests", 72, 4000,
                       "sea", FREQ_MHz, 1200, tf_targets,
                       "p526_vs_p368_telefunken.png",
                       title_suffix="  (BArch RL 19-6/40 ref. 230Q8 App. 2)")

    # Master bar chart
    print("\n[4/4] Master bar chart (all paths)")
    plot_master_bargraph("p526_vs_p368_master_bargraph.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
