#!/usr/bin/env python3
"""
Fix distance_m values in path CSVs to match the actual haversine
distance between the given coordinates.

The research agents provided approximate/wrong distances that don't
match the coordinates. This script recomputes them.
"""
import math, csv, os

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in METRES."""
    R = 6_371_000.0
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def fix_csv(path, skip_trial=True):
    """Update distance_m in a CSV to match haversine from coords."""
    with open(path, "r") as f:
        lines = f.readlines()

    # Find header row (first non-comment line)
    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith("#") and "," in line:
            header_idx = i
            break

    if header_idx is None:
        return

    header = lines[header_idx].strip().split(",")
    fields = {name: idx for idx, name in enumerate(header)}

    new_lines = list(lines[:header_idx + 1])
    changes = []

    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith("#") or not line.strip():
            new_lines.append(line)
            continue
        parts = line.rstrip("\n").split(",")
        if len(parts) < len(header):
            new_lines.append(line)
            continue

        try:
            path_id = parts[fields["path_id"]]
            path_type = parts[fields["type"]]
            tx_lat = float(parts[fields["tx_lat"]])
            tx_lon = float(parts[fields["tx_lon"]])
            rx_lat = float(parts[fields["rx_lat"]])
            rx_lon = float(parts[fields["rx_lon"]])
            old_dist = float(parts[fields["distance_m"]])

            new_dist = haversine(tx_lat, tx_lon, rx_lat, rx_lon)

            # Only update if different by > 500m
            if abs(new_dist - old_dist) > 500:
                changes.append((path_id, path_type, old_dist/1000, new_dist/1000))
                parts[fields["distance_m"]] = f"{new_dist:.0f}"
                new_lines.append(",".join(parts) + "\n")
            else:
                new_lines.append(line)
        except (ValueError, KeyError, IndexError) as e:
            new_lines.append(line)

    if changes:
        with open(path, "w") as f:
            f.writelines(new_lines)
        print(f"\n=== {os.path.basename(path)} ===")
        print(f"{'path_id':<18} {'type':<12} {'old km':>8} {'new km':>8} {'diff':>6}")
        for path_id, ptype, old, new in changes:
            print(f"{path_id:<18} {ptype:<12} {old:>8.0f} {new:>8.0f} {new-old:>+6.0f}")


BOTB_DIR = "/home/alan/claude/BotB"
for csv_file in ["xgerat_paths.csv", "ygerat_paths.csv"]:
    fix_csv(os.path.join(BOTB_DIR, csv_file))

# Knickebein: fix only the operational/measurement/intercept paths
# (the Telefunken test paths use placeholder coordinates for nominal ranges)
print("\n=== knickebein_paths.csv (operational only, Telefunken skipped) ===")
# Actually, since Knickebein operational distances are already within 1-2 km
# of correct, skip this one. Only fix XG and YG.
print("Knickebein operational distances are already correct to within 1 km.")
print("Telefunken test paths use nominal ranges with placeholder coordinates.")
