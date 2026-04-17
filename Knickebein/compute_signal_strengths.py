"""
Compute equisignal field strengths for every Knickebein and Telefunken
path using the canonical post-β-fix botb_itu_analysis.py library.

Outputs a CSV file (botb_signal_strengths.csv) with columns:
  path_name, distance_km, tx_m, rx_m, ground,
  friis_peak_uV, friis_eq_uV,
  sn_peak_uV, sn_eq_uV,
  fock_peak_uV, fock_eq_uV,
  noise_floor_uV,
  fock_eq_snr_dB, status

The CSV uses the same library, same parameters, same CROSSOVER_dB,
and same noise floor as make_p526_vs_p368_graphs.py. Any other script
(e.g. the beam-map script) can read this CSV instead of recomputing
from scratch, guaranteeing consistent numbers across all outputs.

Usage:
    python3 compute_signal_strengths.py
    # → writes botb_signal_strengths.csv to the current directory
    # → also writes a copy to the vault Attachments folder
"""

import sys
import os
import csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))  # parent BotB/ root holds botb_itu_analysis

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
RX_GAIN_DBI = 0.0  # isotropic, matching the graph generator

# Noise floor in μV at 50 Ω (computed from library's N_FLOOR_dBW)
P_NOISE_W = 10.0 ** (N_FLOOR_dBW / 10.0)
V_NOISE_UV = (P_NOISE_W * 50.0) ** 0.5 * 1e6

# SNR → V_rx conversion constant (same as graph generator)
SNR_TO_DBUV = N_FLOOR_dBW + 137.0

# Output paths
BOTB_DIR = _HERE
VAULT_DIR = "/home/alan/Documents/multi_2/Attachments"
CSV_NAME = "botb_signal_strengths.csv"


# ================================================================
#  PATH DEFINITIONS
# ================================================================

# Each entry: (name, distance_km, tx_m, rx_m, ground_type)
# These match knickebein_paths.csv and the Telefunken test table
# in the null doc exactly.

PATHS = [
    # Knickebein operational paths (Kleve Kn-4, TX 111 m, RX 6000 m)
    ("Kleve→Spalding",       440, 111, 6000, "land"),
    ("Kleve→Retford",        512, 111, 6000, "land"),
    ("Kleve→Derby",          529, 111, 6000, "land"),
    ("Kleve→Birmingham",     550, 111, 6000, "land"),

    # Knickebein operational paths (Stollberg Kn-2, TX 72 m, RX 6000 m)
    ("Stollberg→Beeston",    694,  72, 6000, "sea"),
    ("Stollberg→Derby",      711,  72, 6000, "sea"),
    ("Stollberg→Birmingham", 754,  72, 6000, "sea"),
    ("Stollberg→Liverpool",  791,  72, 6000, "sea"),

    # Telefunken July 1939 over-sea range tests
    # (TX 72 m Stollberg-class, RX 4000 m test aircraft, seawater)
    # Source: BArch RL 19-6/40 ref. 230Q8 App. 2
    ("TF_400km",             400,  72, 4000, "sea"),
    ("TF_500km",             500,  72, 4000, "sea"),
    ("TF_700km",             700,  72, 4000, "sea"),
    ("TF_800km",             800,  72, 4000, "sea"),
    ("TF_1000km",           1000,  72, 4000, "sea"),
]


# ================================================================
#  COMPUTATION
# ================================================================

def snr_to_uv(snr_db):
    """Convert SNR (dB above noise floor) to voltage in μV at 50 Ω."""
    return 10.0 ** ((snr_db + SNR_TO_DBUV) / 20.0)


def classify(eq_uv, noise_uv):
    """USABLE / MARGINAL / DEAD classification per the null doc."""
    snr_db = 20.0 * np.log10(eq_uv / noise_uv) if eq_uv > 0 else -999
    if snr_db >= 10.0:
        return "USABLE"
    elif snr_db >= 0.0:
        return "MARGINAL"
    else:
        return "DEAD"


def compute_all():
    """Compute field strengths for every path and return as a list of dicts."""
    results = []

    for name, d_km, tx_m, rx_m, ground in PATHS:
        d_m = d_km * 1000.0

        # ---- Friis flat-Earth (no diffraction loss) ----
        friis = link_budget(d_m, 0.0, FREQ_HZ, RX_GAIN_DBI)
        friis_peak_snr = friis["SNR_dB"]
        friis_eq_snr = friis_peak_snr + CROSSOVER_dB

        # ---- Sommerfeld-Norton flat-Earth ----
        sn_peak_snr = sommerfeld_norton_snr_peak(
            d_km, tx_m, rx_m, ground=ground,
            freq=FREQ_HZ, rx_gain_dBi=RX_GAIN_DBI + 3.0)
        # Note: sn_snr_peak in make_p526_vs_p368_graphs.py uses
        # rx_gain_dBi=3.0 as its default. We match that here so
        # the SN values are consistent with the 4-bar chart.
        sn_eq_snr = sn_peak_snr + CROSSOVER_dB

        # ---- ITU-R P.526-16 Fock globe ----
        itu = itu_diffraction_loss(d_m, tx_m, rx_m, FREQ_HZ,
                                    ground=ground)
        fock = link_budget(d_m, itu["loss_dB"], FREQ_HZ, RX_GAIN_DBI)
        fock_peak_snr = fock["SNR_dB"]
        fock_eq_snr = fock_peak_snr + CROSSOVER_dB

        # ---- Convert all to μV ----
        friis_peak_uv = snr_to_uv(friis_peak_snr)
        friis_eq_uv   = snr_to_uv(friis_eq_snr)
        sn_peak_uv    = snr_to_uv(sn_peak_snr)
        sn_eq_uv      = snr_to_uv(sn_eq_snr)
        fock_peak_uv   = snr_to_uv(fock_peak_snr)
        fock_eq_uv     = snr_to_uv(fock_eq_snr)

        status = classify(fock_eq_uv, V_NOISE_UV)

        results.append({
            "path_name":       name,
            "distance_km":     d_km,
            "tx_m":            tx_m,
            "rx_m":            rx_m,
            "ground":          ground,
            "friis_peak_uV":   friis_peak_uv,
            "friis_eq_uV":     friis_eq_uv,
            "sn_peak_uV":      sn_peak_uv,
            "sn_eq_uV":        sn_eq_uv,
            "fock_peak_uV":    fock_peak_uv,
            "fock_eq_uV":      fock_eq_uv,
            "fock_eq_snr_dB":  fock_eq_snr,
            "noise_floor_uV":  V_NOISE_UV,
            "status":          status,
        })

    return results


# ================================================================
#  OUTPUT
# ================================================================

def write_csv(results, path):
    """Write the results list to a CSV file."""
    fieldnames = [
        "path_name", "distance_km", "tx_m", "rx_m", "ground",
        "friis_peak_uV", "friis_eq_uV",
        "sn_peak_uV", "sn_eq_uV",
        "fock_peak_uV", "fock_eq_uV",
        "fock_eq_snr_dB",
        "noise_floor_uV", "status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Round floats for readability
            out = {}
            for k, v in row.items():
                if isinstance(v, float):
                    if abs(v) >= 1.0:
                        out[k] = f"{v:.4f}"
                    elif abs(v) >= 0.001:
                        out[k] = f"{v:.6f}"
                    else:
                        out[k] = f"{v:.2e}"
                else:
                    out[k] = v
            writer.writerow(out)
    print(f"  saved: {path}")


def main():
    print("=" * 60)
    print("Computing canonical signal strengths")
    print(f"  Library noise floor: {N_FLOOR_dBW:.2f} dBW = {V_NOISE_UV:.4f} μV")
    print(f"  Crossover loss: {CROSSOVER_dB:.1f} dB (5° squint equisignal)")
    print(f"  RX gain: {RX_GAIN_DBI:.1f} dBi (isotropic)")
    print(f"  Frequency: {FREQ_HZ/1e6:.1f} MHz")
    print("=" * 60)
    print()

    results = compute_all()

    # Print summary table
    print(f"{'Path':<25} {'Dist':>5} {'Friis eq':>12} {'SN eq':>12} "
          f"{'Fock eq':>12} {'SNR':>8} {'Status':<10}")
    print("-" * 90)
    for r in results:
        def fmt(v):
            if v >= 1.0:
                return f"{v:.2f} μV"
            elif v >= 0.001:
                return f"{v*1000:.2f} nV"
            else:
                return f"{v:.2e} μV"
        print(f"{r['path_name']:<25} {r['distance_km']:>5} "
              f"{fmt(r['friis_eq_uV']):>12} {fmt(r['sn_eq_uV']):>12} "
              f"{fmt(r['fock_eq_uV']):>12} {r['fock_eq_snr_dB']:>+7.1f} "
              f"{r['status']:<10}")
    print()

    # Write CSVs
    local_path = os.path.join(BOTB_DIR, CSV_NAME)
    vault_path = os.path.join(VAULT_DIR, CSV_NAME)
    write_csv(results, local_path)
    write_csv(results, vault_path)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
