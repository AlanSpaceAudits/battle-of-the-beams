"""
ITU_Calc_ master bar graphs — the canonical, in-sync-with-spreadsheet
comparison of Sommerfeld-Norton flat-Earth (FE, surface reflectivity)
vs ITU-R P.526-16 Fock (GE, globe creeping-wave shadow) for every
Knickebein + Telefunken path.

Four outputs (2 TX perspectives × 2 target batches):
  1. ITU_Calc_master_bargraph_kleve_operational.png
       Kleve 111 m over LAND → Spalding/Retford/Derby/Birmingham
  2. ITU_Calc_master_bargraph_kleve_telefunken.png
       Kleve 111 m over SEA  → TF 392/450/607/696/874 km
  3. ITU_Calc_master_bargraph_stollberg_operational.png
       Stollberg 72 m over SEA → Beeston/Derby/Birmingham/Liverpool
  4. ITU_Calc_master_bargraph_stollberg_telefunken.png
       Stollberg 72 m over SEA → TF 400/500/700/800/1000 km

Kleve ground rule: "land" everywhere except for the Telefunken over-sea
paths (matches the ITU_Calculator sheet's J2/J4/J6 IF-override on tab
`ITU`). Stollberg ground is always "sea".

The first three batches are read straight from botb_signal_strengths.csv
so values are byte-for-byte identical with the spreadsheet. The
Stollberg→TF batch is computed on the fly via botb_itu_analysis because
it is not in the spreadsheet's stock target list.

Two bars per target:
  • Sommerfeld-Norton FE (green, #00E676 — matches the FE wedge on the beam maps)
  • ITU-R P.526-16 Fock   (magenta, #FF1493)
Friis and P.368 GRWAVE are intentionally omitted — per user, the
"direct comparison" is flat-surface-with-reflectivity vs globe creeping
wave only.

Saves to graphs/ and vault Attachments/.
"""

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # parent BotB/ root

from botb_itu_analysis import (
    itu_diffraction_loss,
    link_budget,
    sommerfeld_norton_snr_peak,
    CROSSOVER_dB,
    N_FLOOR_dBW,
)

# ================================================================
#  CONSTANTS
# ================================================================

FREQ_HZ = 31.5e6
RX_GAIN_DBI = 0.0

P_NOISE_W = 10.0 ** (N_FLOOR_dBW / 10.0)
V_NOISE_UV = (P_NOISE_W * 50.0) ** 0.5 * 1e6      # noise floor in μV
SNR_TO_DBUV = N_FLOOR_dBW + 137.0

BOTB_DIR = os.path.join(_HERE, "graphs")
os.makedirs(BOTB_DIR, exist_ok=True)
VAULT_DIR = "/home/alan/Documents/multi_2/Attachments"
CSV_PATH = os.path.join(_HERE, "botb_signal_strengths.csv")

# ---- Style ----
BG_COLOR = "#1a1a1a"
COL_SN    = "#00E676"    # bright green (Sommerfeld-Norton flat) — matches beam-map FE wedge
COL_FOCK  = "#FF1493"    # magenta (ITU-R P.526 Fock globe)
COL_NOISE = "white"


# ================================================================
#  PATHS (for labels + CSV row lookup)
# ================================================================

KLEVE_OPERATIONAL = [
    ("Spalding",   "Kleve→Spalding"),
    ("Retford",    "Kleve→Retford"),
    ("Derby",      "Kleve→Derby"),
    ("Birmingham", "Kleve→Birmingham"),
]

KLEVE_TELEFUNKEN = [
    # display label, CSV lookup key, actual Kleve-to-target km
    ("TF 400 km\n(Kl 392)",  "TF_400km"),
    ("TF 500 km\n(Kl 450)",  "TF_500km"),
    ("TF 700 km\n(Kl 607)",  "TF_700km"),
    ("TF 800 km\n(Kl 696)",  "TF_800km"),
    ("TF 1000 km\n(Kl 874)", "TF_1000km"),
]

STOLL_OPERATIONAL = [
    ("Beeston",    "Stollberg→Beeston"),
    ("Derby",      "Stollberg→Derby"),
    ("Birmingham", "Stollberg→Birmingham"),
    ("Liverpool",  "Stollberg→Liverpool"),
]

# Stollberg→TF entries are computed live because they're not in the CSV.
# The TF test flights' nominal range IS the Stollberg-to-target distance
# (westward from Stollberg), so these are 400/500/700/800/1000 km directly.
STOLL_TELEFUNKEN_PARAMS = [
    # display label, distance_km (= Stollberg-to-target), tx_m, rx_m, ground
    ("TF 400 km",  400, 72, 4000, "sea"),
    ("TF 500 km",  500, 72, 4000, "sea"),
    ("TF 700 km",  700, 72, 4000, "sea"),
    ("TF 800 km",  800, 72, 4000, "sea"),
    ("TF 1000 km", 1000, 72, 4000, "sea"),
]


# ================================================================
#  HELPERS
# ================================================================

def snr_to_uv(snr_db):
    """SNR (dB above noise floor) → V_rx(μV) at 50 Ω."""
    return 10.0 ** ((snr_db + SNR_TO_DBUV) / 20.0)


def load_csv_rows():
    """Load botb_signal_strengths.csv into a {path_name: row} dict."""
    rows = {}
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["path_name"]] = row
    return rows


def compute_stollberg_tf():
    """Compute SN + Fock equisignal μV for Stollberg→TF, on the fly,
    using the exact same library calls as compute_signal_strengths.py."""
    out = {}
    for label, d_km, tx_m, rx_m, ground in STOLL_TELEFUNKEN_PARAMS:
        d_m = d_km * 1000.0
        # Sommerfeld-Norton flat-Earth peak SNR, then +CROSSOVER_dB for eq
        sn_peak = sommerfeld_norton_snr_peak(
            d_km, tx_m, rx_m, ground=ground,
            freq=FREQ_HZ, rx_gain_dBi=RX_GAIN_DBI)
        sn_eq = sn_peak + CROSSOVER_dB
        # ITU-R P.526-16 Fock
        itu = itu_diffraction_loss(d_m, tx_m, rx_m, FREQ_HZ, ground=ground)
        fock_peak = link_budget(d_m, itu["loss_dB"], FREQ_HZ, RX_GAIN_DBI)["SNR_dB"]
        fock_eq = fock_peak + CROSSOVER_dB
        out[label] = {
            "sn_eq_uV":   snr_to_uv(sn_eq),
            "fock_eq_uV": snr_to_uv(fock_eq),
            "distance_km": d_km,
        }
    return out


# ================================================================
#  PLOTTING
# ================================================================

def plot_bars(labels, sn_vals, fock_vals, title, subtitle, fname):
    """Side-by-side SN + Fock bar chart on a μV log axis."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_yscale("log")

    x = np.arange(len(labels))
    w = 0.36
    y_lim = (1e-6, 1e5)
    bar_bottom = y_lim[0]

    bars_sn = ax.bar(x - w / 2, sn_vals, w, bottom=bar_bottom,
                     color=COL_SN, alpha=0.92,
                     label="Sommerfeld-Norton FE (flat + surface reflectivity)",
                     edgecolor="white", linewidth=0.5, zorder=3)
    bars_fk = ax.bar(x + w / 2, fock_vals, w, bottom=bar_bottom,
                     color=COL_FOCK, alpha=0.92,
                     label="ITU-R P.526-16 Fock (globe creeping-wave shadow)",
                     edgecolor="white", linewidth=0.5, zorder=3)

    noise_line = ax.axhline(y=V_NOISE_UV, color=COL_NOISE, linewidth=1.2,
                            linestyle="-", zorder=2,
                            label=f"Noise floor ({V_NOISE_UV:.2f} μV @ 50 Ω)")

    # Value labels
    for bars, vals in ((bars_sn, sn_vals), (bars_fk, fock_vals)):
        for bar, val in zip(bars, vals):
            y_pos = max(val * 1.25, y_lim[0] * 1.5)
            if val >= 1000:
                txt = f"{val:,.0f}"
            elif val >= 10:
                txt = f"{val:.1f}"
            elif val >= 1:
                txt = f"{val:.2f}"
            elif val >= 0.001:
                txt = f"{val:.3f}"
            else:
                txt = f"{val:.1e}"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, txt,
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="white", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color="white")
    ax.set_ylabel("Equisignal voltage at receiver input (μV into 50 Ω, 5° squint)",
                  fontsize=11, color="white")
    ax.set_ylim(*y_lim)

    def fmt_uv(val, pos):
        if val >= 1:
            return f"{val:g} μV"
        if val >= 1e-3:
            return f"{val*1e3:g} nV"
        return f"{val:.0e} μV"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_uv))

    ax.set_title(f"{title}\n{subtitle}",
                 fontsize=13, fontweight="bold", color="white", pad=16)
    ax.grid(axis="y", alpha=0.22, color="gray", zorder=0, which="both")
    ax.tick_params(colors="white")

    ax.legend([bars_sn, bars_fk, noise_line],
              [bars_sn.get_label(), bars_fk.get_label(), noise_line.get_label()],
              loc="upper right", fontsize=10, framealpha=0.6,
              facecolor=BG_COLOR, edgecolor="#444444", labelcolor="white")

    fig.tight_layout()

    for outdir in (BOTB_DIR, VAULT_DIR):
        outpath = os.path.join(outdir, fname)
        fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"  Saved: {outpath}")
    plt.close(fig)


# ================================================================
#  STATION SWEEP (distance vs equisignal μV)
# ================================================================

def _radio_horizon_km(tx_m, rx_m):
    """4/3-Earth radio horizon from both ends, in km (heights in metres)."""
    return (17.0 * tx_m) ** 0.5 + (17.0 * rx_m) ** 0.5


def plot_itu_calc_sweep(title, subtitle, tx_m, rx_m, ground, d_max_km,
                         targets, fname):
    """Distance sweep of SN and Fock equisignal μV, for a single TX/RX/ground.

    targets: list of (label, distance_km) for vertical dotted markers.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 7.5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_yscale("log")

    # ---- Sweep ----
    d_km = np.arange(50, d_max_km + 1, 10.0)
    sn_uv = np.zeros_like(d_km)
    fk_uv = np.zeros_like(d_km)
    for i, d in enumerate(d_km):
        d_m = float(d) * 1000.0
        sn_peak = sommerfeld_norton_snr_peak(
            float(d), tx_m, rx_m, ground=ground,
            freq=FREQ_HZ, rx_gain_dBi=RX_GAIN_DBI)
        itu = itu_diffraction_loss(d_m, tx_m, rx_m, FREQ_HZ, ground=ground)
        fk_peak = link_budget(d_m, itu["loss_dB"], FREQ_HZ, RX_GAIN_DBI)["SNR_dB"]
        sn_uv[i] = snr_to_uv(sn_peak + CROSSOVER_dB)
        fk_uv[i] = snr_to_uv(fk_peak + CROSSOVER_dB)

    ax.plot(d_km, sn_uv, color=COL_SN, linewidth=2.4,
            label="Sommerfeld-Norton FE (flat + surface reflectivity)",
            zorder=5)
    ax.plot(d_km, fk_uv, color=COL_FOCK, linewidth=2.4,
            label="ITU-R P.526-16 Fock (globe creeping-wave shadow)",
            zorder=5)

    # ---- Reference lines ----
    ax.axhline(V_NOISE_UV, color=COL_NOISE, linewidth=1.2, linestyle="-",
               label=f"Noise floor ({V_NOISE_UV:.2f} μV @ 50 Ω)", zorder=3)

    horizon_km = _radio_horizon_km(tx_m, rx_m)
    ax.axvline(horizon_km, color="#9AA0B0", linewidth=1.2, linestyle=(0, (6, 4)),
               alpha=0.85, zorder=3,
               label=f"Radio horizon ({horizon_km:.0f} km, 4/3 Earth)")

    # ---- Target markers ----
    # Targets can be (label, distance_km) or (label, distance_km, row). row
    # is an integer stagger index: row 0 at the bottom, each extra row
    # raises the label by two decades on the log axis so clustered targets
    # (Kleve's Retford/Derby/Birmingham within 40 km) don't overlap.
    y_lim = (1e-6, 1e5)
    for t in targets:
        if len(t) == 3:
            lbl, d, row = t
        else:
            lbl, d = t
            row = 0
        ax.axvline(d, color="#90C4E0", linewidth=1.0, linestyle=":",
                   alpha=0.75, zorder=2)
        y_pos = y_lim[0] * 4 * (100.0 ** row)
        ax.text(d, y_pos, f"{lbl}\n{d:.0f} km",
                color="#90C4E0", fontsize=9, fontweight="bold",
                ha="center", va="bottom", rotation=90,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#0a0e1c",
                          edgecolor="#3a4258", linewidth=0.8, alpha=0.88))

    # ---- Axes ----
    ax.set_xlim(d_km[0], d_max_km)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("Distance (km)", fontsize=11, color="white")
    ax.set_ylabel("Equisignal voltage at receiver input "
                  "(μV into 50 Ω, 5° squint)", fontsize=11, color="white")

    def fmt_uv(val, pos):
        if val >= 1:
            return f"{val:g} μV"
        if val >= 1e-3:
            return f"{val*1e3:g} nV"
        return f"{val:.0e} μV"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_uv))

    ax.set_title(f"{title}\n{subtitle}",
                 fontsize=13, fontweight="bold", color="white", pad=14)
    ax.grid(which="both", alpha=0.2, color="gray", zorder=0)
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.6,
              facecolor=BG_COLOR, edgecolor="#444444", labelcolor="white",
              ncol=1)
    fig.tight_layout()

    for outdir in (BOTB_DIR, VAULT_DIR):
        outpath = os.path.join(outdir, fname)
        fig.savefig(outpath, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"  Saved: {outpath}")
    plt.close(fig)


# ================================================================
#  MAIN
# ================================================================

def main():
    print("=" * 68)
    print("Generating ITU_Calc_ master bar graphs (SN vs Fock, μV)")
    print(f"  CSV source:  {CSV_PATH}")
    print(f"  Noise floor: {V_NOISE_UV:.4f} μV at 50 Ω")
    print(f"  CROSSOVER:   {CROSSOVER_dB:.2f} dB")
    print("=" * 68)

    csv_rows = load_csv_rows()

    # ---- 1. Kleve operational (land) ----
    labels, sn, fk = [], [], []
    for disp, key in KLEVE_OPERATIONAL:
        r = csv_rows[key]
        d_km = int(float(r["distance_km"]))
        labels.append(f"{disp}\n{d_km} km")
        sn.append(float(r["sn_eq_uV"]))
        fk.append(float(r["fock_eq_uV"]))
    plot_bars(np.array(labels), np.array(sn), np.array(fk),
              title="Kleve → operational UK targets",
              subtitle="TX 111 m, overland path, 31.5 MHz, 3 kW — "
                       "Sommerfeld-Norton flat-Earth vs ITU-R P.526 Fock globe",
              fname="ITU_Calc_master_bargraph_kleve_operational.png")

    # ---- 2. Kleve → Telefunken (over sea) ----
    labels, sn, fk = [], [], []
    for disp, key in KLEVE_TELEFUNKEN:
        r = csv_rows[key]
        labels.append(disp)
        sn.append(float(r["sn_eq_uV"]))
        fk.append(float(r["fock_eq_uV"]))
    plot_bars(np.array(labels), np.array(sn), np.array(fk),
              title="Kleve → Telefunken 1939 over-sea test ranges",
              subtitle="TX 111 m over SEA (Kleve→target Haversine km in parens) — "
                       "Sommerfeld-Norton flat-Earth vs ITU-R P.526 Fock globe",
              fname="ITU_Calc_master_bargraph_kleve_telefunken.png")

    # ---- 3. Stollberg operational (sea) ----
    labels, sn, fk = [], [], []
    for disp, key in STOLL_OPERATIONAL:
        r = csv_rows[key]
        d_km = int(float(r["distance_km"]))
        labels.append(f"{disp}\n{d_km} km")
        sn.append(float(r["sn_eq_uV"]))
        fk.append(float(r["fock_eq_uV"]))
    plot_bars(np.array(labels), np.array(sn), np.array(fk),
              title="Stollberg → operational UK targets",
              subtitle="TX 72 m over sea, 31.5 MHz, 3 kW — "
                       "Sommerfeld-Norton flat-Earth vs ITU-R P.526 Fock globe",
              fname="ITU_Calc_master_bargraph_stollberg_operational.png")

    # ---- 4. Stollberg → Telefunken (computed live) ----
    stoll_tf = compute_stollberg_tf()
    labels, sn, fk = [], [], []
    for disp, d_km, *_ in STOLL_TELEFUNKEN_PARAMS:
        r = stoll_tf[disp]
        labels.append(disp)
        sn.append(r["sn_eq_uV"])
        fk.append(r["fock_eq_uV"])
    plot_bars(np.array(labels), np.array(sn), np.array(fk),
              title="Stollberg → Telefunken 1939 over-sea test ranges",
              subtitle="TX 72 m over sea — Stollberg-to-target range per 1939 test log — "
                       "Sommerfeld-Norton flat-Earth vs ITU-R P.526 Fock globe",
              fname="ITU_Calc_master_bargraph_stollberg_telefunken.png")

    # ================================================================
    #  Distance sweeps — SN vs Fock over distance (μV log axis)
    # ================================================================

    # Kleve → UK Midlands sweep (land, 111 m TX, 6 km RX).
    # Retford/Derby/Birmingham sit within 40 km of each other, so stagger
    # them across rows 0/1/2 to keep the labels legible.
    plot_itu_calc_sweep(
        title="Kleve → Midlands",
        subtitle="TX 111 m   |   RX 6,000 m   |   31.5 MHz, 3 kW, 26 dBi   |   overland",
        tx_m=111, rx_m=6000, ground="land", d_max_km=900,
        targets=[("Spalding", 440, 0), ("Retford", 512, 0),
                 ("Derby", 530, 1), ("Birmingham", 550, 0)],
        fname="ITU_Calc_sweep_kleve_operational.png")

    # Stollberg → UK Midlands sweep (sea, 72 m TX, 6 km RX)
    plot_itu_calc_sweep(
        title="Stollberg → Midlands",
        subtitle="TX 72 m   |   RX 6,000 m   |   31.5 MHz, 3 kW, 26 dBi   |   seawater",
        tx_m=72, rx_m=6000, ground="sea", d_max_km=1000,
        targets=[("Beeston", 694, 1), ("Derby", 711, 0),
                 ("Birmingham", 754, 0), ("Liverpool", 791, 0)],
        fname="ITU_Calc_sweep_stollberg_operational.png")

    # Telefunken July 1939 over-sea tests (72 m TX, 4 km RX, sea)
    plot_itu_calc_sweep(
        title="Telefunken July 1939 over-sea tests",
        subtitle="TX 72 m   |   RX 4,000 m   |   31.5 MHz, 3 kW, 26 dBi   |   seawater"
                 "   (BArch RL 19-6/40 ref. 230Q8 App. 2)",
        tx_m=72, rx_m=4000, ground="sea", d_max_km=1200,
        targets=[("TF", 400), ("TF", 500), ("TF", 700),
                 ("TF", 800), ("TF", 1000)],
        fname="ITU_Calc_sweep_telefunken.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
