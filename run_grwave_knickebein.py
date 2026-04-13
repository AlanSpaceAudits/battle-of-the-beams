"""
Run ITU P.368 GRWAVE ground-wave propagation calculator for the Knickebein
paths. This directly tests DC Dan's claim that ground-wave propagation rescues
the beams on a globe.

Note: grwave is officially valid for 10 kHz to 30 MHz. Knickebein at 31.5 MHz
is 1.5 MHz above the stated range. We run at 30 MHz (highest valid) as a
best-case for ground-wave amplitude, since ground-wave attenuation increases
sharply with frequency above a few MHz over any real ground.

Outputs field strength in dB(uV/m) and compares against the FuBl 2 receiver
threshold.
"""

import grwave
import numpy as np


# ================================================================
#  PATHS from knickebein_paths.csv (verified Kleve, Stollberg paths)
# ================================================================
# TX altitudes are terrain + antenna centre per Trenkle (1979) via christianch.ch
# RX altitude is He 111 operational (6000 m)

KN_PATHS = [
    # name,           d_km,  tx_m, rx_m, notes
    ("Kleve→Spalding",     440, 111, 6000, "Bufton measurement, 21 Jun 1940"),
    ("Kleve→Retford",      512, 111, 6000, "Enigma intercept coords"),
    ("Kleve→Derby",        529, 111, 6000, "Rolls-Royce Merlin works"),
    ("Kleve→Birmingham",   550, 111, 6000, ""),
    ("Stollberg→Beeston",  694, 72,  6000, "Cross-beam target"),
    ("Stollberg→Derby",    711, 72,  6000, ""),
    ("Stollberg→Birmingham", 754, 72, 6000, ""),
    ("Stollberg→Liverpool",  791, 72, 6000, ""),
]

# Telefunken 1939 over-sea tests (sea path, 4000 m altitude)
TF_PATHS = [
    ("Telefunken 400 km",  400, 72, 4000, "FuBl 1 rod"),
    ("Telefunken 500 km",  500, 72, 4000, "FuBl 1 trailing wire"),
    ("Telefunken 700 km",  700, 72, 4000, "FuBl + selectivity rod"),
    ("Telefunken 800 km",  800, 72, 4000, "FuBl + selectivity wire"),
    ("Telefunken 1000 km", 1000, 72, 4000, "special RX xdip"),
]

# Ground parameter sets (ITU-R P.527)
GROUND_TYPES = {
    "seawater":      {"sigma": 5.0,   "epslon": 70.0},
    "wet_ground":    {"sigma": 1e-2,  "epslon": 30.0},
    "average_land":  {"sigma": 5e-3,  "epslon": 15.0},
    "dry_ground":    {"sigma": 1e-3,  "epslon": 4.0},
}

# Knickebein parameters
FREQ_MHz = 30.0            # grwave upper limit (actual Knickebein = 31.5)
TX_POWER_W = 3000          # Large Knickebein Telefunken spec

# Receiver sensitivity threshold for an AGC-locked audible signal.
# ITU-R P.372 galactic noise at 30 MHz is Fa ≈ 18 dB.
# kTB for 1 kHz IF BW at 290 K is -173 dBW + 30 = -143 dBW at 1 Hz.
# For 6 kHz AM voice BW: Noise = -204 + 18 + 10*log10(6000) = -204 + 18 + 37.8 = -148.2 dBW
# Convert to field strength E_min (dBuV/m) at receiving antenna:
# For isotropic RX (G=0 dBi) at 30 MHz, E (dBuV/m) = P(dBW) + 77.2 + 20*log10(f_MHz)
# = -148.2 + 77.2 + 29.5 = -41.5 dBuV/m is the noise floor.
# Detection floor is +10 dB above that:
NOISE_dBuVm = -41.5
DETECT_dBuVm = NOISE_dBuVm + 10.0   # -31.5 dBuV/m


def run_path(name, d_km, tx_m, rx_m, ground_name, notes=""):
    """Run grwave for a single path."""
    gt = GROUND_TYPES[ground_name]
    params = {
        "freqMHz": FREQ_MHz,
        "sigma": gt["sigma"],
        "epslon": gt["epslon"],
        "dmax": max(d_km + 10, 50),
        "hrr": rx_m,
        "htt": tx_m,
        "dstep": 20,
        "txwatt": TX_POWER_W,
    }
    try:
        data = grwave.grwave(params)
    except Exception as e:
        return None, f"ERROR: {e}"

    # grwave returns rows at DMIN, DMIN+DSTEP, ... up to DMAX.
    # Find the row closest to d_km.
    # NOTE: Despite the README saying "mV/m", the actual fs column from
    # grwave.for is already in **dB(uV/m)**. Verified by direct inspection of
    # raw output and by the presence of large negative values in the residue
    # series (Fock) region.
    if len(data) == 0:
        return None, "empty output"
    distances = data.index.values.astype(float)
    idx = np.argmin(np.abs(distances - d_km))
    fs_dBuVm = float(data.iloc[idx]["fs"])
    pathloss_dB = float(data.iloc[idx]["pathloss"])
    return {
        "name": name,
        "d_km": d_km,
        "tx_m": tx_m,
        "rx_m": rx_m,
        "ground": ground_name,
        "fs_dBuVm": fs_dBuVm,
        "pathloss_dB": pathloss_dB,
        "above_noise_dB": fs_dBuVm - NOISE_dBuVm,
        "above_detect_dB": fs_dBuVm - DETECT_dBuVm,
        "notes": notes,
    }, None


def print_header(label):
    print()
    print("=" * 100)
    print(f"  {label}")
    print(f"  ITU-R P.368 GRWAVE ground-wave propagation, {FREQ_MHz} MHz, "
          f"TX {TX_POWER_W} W")
    print(f"  Detection floor (+10 dB above galactic noise): "
          f"{DETECT_dBuVm:.1f} dBuV/m")
    print("=" * 100)
    print(f"  {'Path':<25} {'Dist':>5} {'TX':>4} {'RX':>5} {'Ground':<12} "
          f"{'FS':>8} {'SNR':>6} {'vs Det':>8} {'Status':<10}")
    print(f"  {'':25} {'km':>5} {'m':>4} {'m':>5} {'type':<12} "
          f"{'dBuV/m':>8} {'dB':>6} {'dB':>8}")
    print("-" * 100)


def print_row(r):
    if r is None:
        return
    if r["above_detect_dB"] >= 0:
        status = "USABLE"
    else:
        status = "UNUSABLE"
    print(f"  {r['name']:<25} {r['d_km']:>5} {r['tx_m']:>4} {r['rx_m']:>5} "
          f"{r['ground']:<12} "
          f"{r['fs_dBuVm']:>8.1f} {r['above_noise_dB']:>6.1f} "
          f"{r['above_detect_dB']:>+8.1f} {status:<10}")


def main():
    # Kleve: continental land path (mixed land/sea impossible without custom
    # mixed-path calc, so we bracket with sea best-case and dry ground
    # worst-case).
    print_header("KLEVE → MIDLANDS (continental land path bracket)")
    for name, d_km, tx, rx, notes in KN_PATHS[:4]:
        for ground in ["seawater", "average_land", "dry_ground"]:
            r, err = run_path(name, d_km, tx, rx, ground, notes)
            if err:
                print(f"  {name:<25} {d_km:>5}                 {ground:<12} "
                      f"{err}")
            else:
                print_row(r)
        print()

    print_header("STOLLBERG → MIDLANDS (mixed sea + land path)")
    for name, d_km, tx, rx, notes in KN_PATHS[4:]:
        for ground in ["seawater", "average_land", "dry_ground"]:
            r, err = run_path(name, d_km, tx, rx, ground, notes)
            if err:
                print(f"  {name:<25} {d_km:>5}                 {ground:<12} "
                      f"{err}")
            else:
                print_row(r)
        print()

    print_header("TELEFUNKEN JULY 1939 OVER-SEA TESTS (seawater ground)")
    for name, d_km, tx, rx, notes in TF_PATHS:
        r, err = run_path(name, d_km, tx, rx, "seawater", notes)
        if err:
            print(f"  {name:<25} {d_km:>5}                 seawater     {err}")
        else:
            print_row(r)


if __name__ == "__main__":
    main()
