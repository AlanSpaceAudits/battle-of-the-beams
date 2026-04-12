#!/usr/bin/env python3
"""
Run ITU-R P.526 analysis on all three German beam systems:
  - Knickebein (31.5 MHz)
  - X-Gerät (66-75 MHz)
  - Y-Gerät (42-48 MHz)

Generates per-path graphs (full, equisignal-only) for each system.
"""

from botb_itu_analysis import (
    analyse_all_paths, print_results,
    generate_per_path_graphs, generate_per_path_graphs_equisignal_only,
)

systems = [
    ("Knickebein (31.5 MHz)",  "knickebein_paths.csv", "kb",   "kbeq"),
    ("X-Gerät (66-75 MHz)",    "xgerat_paths.csv",     "xg",   "xgeq"),
    ("Y-Gerät (42-48 MHz)",    "ygerat_paths.csv",     "yg",   "ygeq"),
]

for name, csv, pk_prefix, eq_prefix in systems:
    print("\n" + "=" * 90)
    print(f"  {name}  |  {csv}")
    print("=" * 90)

    results = analyse_all_paths(csv_name=csv)
    print_results(results)

    print(f"\nGenerating {name} per-path graphs...")
    generate_per_path_graphs(results, prefix=pk_prefix)
    generate_per_path_graphs_equisignal_only(results, prefix=eq_prefix)

print("\nAll systems complete.")
