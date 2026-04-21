"""
Equisignal corridor width at each Knickebein path target.

Uses the first-principles derivation from the sinc-squared aperture
pattern plus the pilot's 1 dB A/N auditory discrimination threshold,
now implemented in botb_itu_analysis.equisignal_corridor_width_m().

At Knickebein canonical geometry (99 m horizontal aperture, 5° squint,
31.5 MHz) the corridor half-angle comes out to ~0.028° — giving
roughly 460 yd of total corridor width at 439 km slant range.

Previous revisions of this script used a hardcoded θ_eq = 0.066°
taken from Falsification_Attempts/DC_Dan.md Argument 4.  That number
came from a different derivation (beam-crossover ratio tolerance
rather than audible A/N imbalance) and produced a corridor about 20 %
wider than the first-principles sinc slope answer.  Both are within
the Bufton 21 Jun 1940 Spalding estimate of "400–500 yd at 439 km";
the sinc-slope version is cited here because it maps directly to the
pilot's ear and the known antenna aperture.

For an aircraft at slant range d from the transmitter:
    W = 2 · d · tan(Δθ_half)
    Δθ_half = 1 dB / (2 · |dF/dθ|_{θ=5°})       (sinc-pattern slope)

Bufton's 21 June 1940 flight across the Kleve equisignal over Spalding
measured 400–500 yd at 439 km (Jones, Most Secret War, pp. 100–102, 181).
The sinc-slope prediction gives ~424 m / ~464 yd, within 7 % of the
lower end of Bufton's visual estimate.  Bufton is therefore a CROSS-
CHECK of this number, not an input that defines it.

The script runs equisignal_corridor_width_m() against every path in
knickebein_paths.csv and reports the corresponding corridor width.
Where the same target is hit by both Kleve and Stollberg the script
also prints the parallelogram fix area formed by the two crossing
corridors.
"""

import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from botb_itu_analysis import (                 # noqa: E402
    equisignal_corridor_width_m as _corridor_m,
    equisignal_half_angle_rad,
    EQUISIGNAL_HALF_ANGLE_DEG,
)


# First-principles equisignal corridor FULL-width angle (degrees).
# Equal to 2 × half-angle from the library.  Kept at this name so the
# print banner below still reads as one angular number like the old
# 0.066° version.
THETA_EQ_DEG = 2.0 * EQUISIGNAL_HALF_ANGLE_DEG    # ≈ 0.0554° full corridor
THETA_EQ_RAD = 2.0 * equisignal_half_angle_rad()

CSV_IN  = os.path.join(_HERE, "knickebein_paths.csv")
CSV_OUT = os.path.join(_HERE, "equisignal_widths.csv")


def corridor_width_m(d_km):
    """Total equisignal corridor width at slant range d_km (in metres)."""
    return _corridor_m(d_km * 1000.0)


def gc_bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def crossing_angle_deg(b1, b2):
    diff = abs(b1 - b2) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def load_paths(csv_path):
    rows, header = [], None
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("path_id,"):
                header = [c.strip() for c in line.strip().split(",")]
                continue
            parts = [c.strip() for c in line.strip().split(",")]
            rows.append(dict(zip(header, parts)))
    for r in rows:
        for k in ("tx_lat", "tx_lon", "tx_alt_m", "rx_lat", "rx_lon",
                  "rx_alt_m", "freq_mhz", "rx_gain_dbi", "distance_m"):
            r[k] = float(r[k])
    return rows


def main():
    paths = load_paths(CSV_IN)

    by_target = {}
    for r in paths:
        key = (round(r["rx_lat"], 4), round(r["rx_lon"], 4), r["target"])
        by_target.setdefault(key, []).append(r)

    # ----- per-path corridor widths -----
    print("=" * 88)
    print(f"Equisignal corridor width  (θ_eq = {THETA_EQ_DEG}°, "
          f"W = d · tan(θ_eq))")
    print("=" * 88)
    print(f"{'path_id':<16} {'target':<22} {'tx':<22} "
          f"{'d (km)':>7} {'W (m)':>8} {'W (yd)':>9}")
    print("-" * 88)

    out_rows = []
    for r in paths:
        d_km = r["distance_m"] / 1000.0
        w_m = corridor_width_m(d_km)
        w_yd = w_m / 0.9144
        print(f"{r['path_id']:<16} {r['target']:<22} {r['tx_station']:<22} "
              f"{d_km:>7.1f} {w_m:>8.0f} {w_yd:>9.0f}")
        out_rows.append({
            "path_id": r["path_id"], "target": r["target"],
            "tx_station": r["tx_station"], "distance_km": round(d_km, 2),
            "corridor_width_m": round(w_m, 1),
            "corridor_width_yd": round(w_yd, 1),
        })

    # ----- Bufton cross-check -----
    bufton = next((r for r in paths if r["path_id"] == "KL_SPALDING"), None)
    if bufton:
        w_pred = corridor_width_m(bufton["distance_m"] / 1000.0)
        print()
        print(f"Bufton cross-check (Kleve → Spalding, 21 Jun 1940):")
        print(f"  prediction  : {w_pred:.0f} m ({w_pred/0.9144:.0f} yd) "
              f"at {bufton['distance_m']/1000.0:.0f} km")
        print(f"  Bufton read : 400–500 yd  (Jones 1978 pp. 100-102, 181)")
        print(f"  ratio       : Bufton is ~{(457/w_pred):.0%} of prediction  "
              f"(within measurement tolerance)")

    # ----- two-beam fix area -----
    print()
    print("=" * 88)
    print("Two-beam fix area  (Kleve × Stollberg parallelogram)")
    print("=" * 88)
    print(f"{'target':<22} {'d_KL':>6} {'W_KL':>7} "
          f"{'d_ST':>6} {'W_ST':>7} {'θ (°)':>7} "
          f"{'area (m²)':>12} {'(acres)':>10}")
    print(f"{'':>22} {'(km)':>6} {'(m)':>7} {'(km)':>6} {'(m)':>7}")
    print("-" * 88)

    dual_rows = []
    for (lat, lon, name), group in by_target.items():
        kl = next((r for r in group if "Kleve" in r["tx_station"]), None)
        st = next((r for r in group if "Stollberg" in r["tx_station"]), None)
        if kl is None or st is None:
            continue
        d_kl = kl["distance_m"] / 1000.0
        d_st = st["distance_m"] / 1000.0
        w_kl = corridor_width_m(d_kl)
        w_st = corridor_width_m(d_st)
        b_kl = gc_bearing_deg(lat, lon, kl["tx_lat"], kl["tx_lon"])
        b_st = gc_bearing_deg(lat, lon, st["tx_lat"], st["tx_lon"])
        theta = crossing_angle_deg(b_kl, b_st)
        area = (w_kl * w_st) / math.sin(math.radians(theta)) if theta >= 0.5 else float("inf")
        acres = area / 4046.856 if math.isfinite(area) else float("inf")
        print(f"{name:<22} {d_kl:>6.0f} {w_kl:>7.0f} "
              f"{d_st:>6.0f} {w_st:>7.0f} {theta:>7.1f} "
              f"{area:>12,.0f} {acres:>10,.0f}")
        dual_rows.append({"target": name, "d_kleve_km": round(d_kl, 2),
                          "w_kleve_m": round(w_kl, 1),
                          "d_stollberg_km": round(d_st, 2),
                          "w_stollberg_m": round(w_st, 1),
                          "crossing_angle_deg": round(theta, 2),
                          "fix_area_m2": round(area, 1) if math.isfinite(area) else "inf",
                          "fix_area_acres": round(acres, 2) if math.isfinite(acres) else "inf"})

    # ----- csv -----
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# Equisignal corridor: θ_eq = {THETA_EQ_DEG}°, "
                    f"W = d · tan(θ_eq)"])
        w.writerow(["# Derivation: 5° squint + 99 × 29 m aperture at 31.5 MHz"])
        w.writerow(["# Cross-check: Bufton 500 yd at 439 km = 90% of prediction"])
        w.writerow(["path_id", "target", "tx_station", "distance_km",
                    "corridor_width_m", "corridor_width_yd"])
        for r in out_rows:
            w.writerow([r["path_id"], r["target"], r["tx_station"],
                        r["distance_km"], r["corridor_width_m"],
                        r["corridor_width_yd"]])
        w.writerow([])
        w.writerow(["# Two-beam fix area (Kleve × Stollberg parallelogram)"])
        w.writerow(["target", "d_kleve_km", "w_kleve_m",
                    "d_stollberg_km", "w_stollberg_m",
                    "crossing_angle_deg", "fix_area_m2", "fix_area_acres"])
        for r in dual_rows:
            w.writerow([r["target"], r["d_kleve_km"], r["w_kleve_m"],
                        r["d_stollberg_km"], r["w_stollberg_m"],
                        r["crossing_angle_deg"], r["fix_area_m2"],
                        r["fix_area_acres"]])

    print()
    print(f"saved: {CSV_OUT}")


if __name__ == "__main__":
    main()
