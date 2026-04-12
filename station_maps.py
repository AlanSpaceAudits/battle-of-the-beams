#!/usr/bin/env python3
"""
Station map plots for the Battle of the Beams null hypothesis.

Generates three maps, one per system, showing the two operationally
significant station pairs with their beam paths to confirmed targets.

  Knickebein: Stollberg + Kleve -> Derby (Rolls-Royce Merlin works)
  X-Gerät:    Cotentin + Audembert -> Coventry (Moonlight Sonata)
  Y-Gerät:    Cassel + Beaumont-Hague -> Liverpool/Birkenhead

Uses Natural Earth country polygons from a GeoJSON file for the
coastlines. No cartopy required.
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon
from matplotlib.collections import PatchCollection

# ================================================================
#  LOAD COUNTRY POLYGONS
# ================================================================
GEOJSON_PATH = "/tmp/countries.geojson"

with open(GEOJSON_PATH, "r") as f:
    countries = json.load(f)

def draw_countries(ax, countries_geojson, lon_range, lat_range,
                   fill_color='#2a2a2a', edge_color='#666666',
                   linewidth=0.5):
    """Draw country polygons onto the given axes."""
    for feature in countries_geojson["features"]:
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            polys = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue

        for poly in polys:
            # Each poly is [outer_ring, inner_ring1, ...]
            outer = np.array(poly[0])
            # Cull to visible bounds for speed
            lons = outer[:, 0]
            lats = outer[:, 1]
            if (lons.max() < lon_range[0] or lons.min() > lon_range[1] or
                lats.max() < lat_range[0] or lats.min() > lat_range[1]):
                continue
            patch = mplPolygon(outer, closed=True,
                               facecolor=fill_color,
                               edgecolor=edge_color,
                               linewidth=linewidth)
            ax.add_patch(patch)

# ================================================================
#  GREAT CIRCLE UTILITY
# ================================================================
def great_circle_points(lat1, lon1, lat2, lon2, n=50):
    """Generate n points along the great circle from (lat1,lon1) to
    (lat2,lon2). Returns (lats, lons) arrays. Uses spherical
    interpolation."""
    # Convert to radians
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)

    # Compute angular distance
    d = 2 * math.asin(math.sqrt(
        math.sin((lat2r - lat1r) / 2) ** 2 +
        math.cos(lat1r) * math.cos(lat2r) *
        math.sin((lon2r - lon1r) / 2) ** 2
    ))

    if d < 1e-10:
        return np.array([lat1]), np.array([lon1])

    # Interpolate
    lats = np.zeros(n)
    lons = np.zeros(n)
    for i in range(n):
        f = i / (n - 1)
        A = math.sin((1 - f) * d) / math.sin(d)
        B = math.sin(f * d) / math.sin(d)
        x = A * math.cos(lat1r) * math.cos(lon1r) + B * math.cos(lat2r) * math.cos(lon2r)
        y = A * math.cos(lat1r) * math.sin(lon1r) + B * math.cos(lat2r) * math.sin(lon2r)
        z = A * math.sin(lat1r) + B * math.sin(lat2r)
        lats[i] = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
        lons[i] = math.degrees(math.atan2(y, x))
    return lats, lons


# ================================================================
#  MAP GENERATION
# ================================================================
def make_system_map(title, subtitle, stations, target, bounds, filename,
                    station_color='#F44336', target_color='#90C4E0',
                    line_color='#FF1493'):
    """
    Generate a system map.

    Parameters
    ----------
    title : str
    subtitle : str
    stations : list of (name, lat, lon, designation)
    target : (name, lat, lon)
    bounds : (lon_min, lon_max, lat_min, lat_max)
    filename : output PNG file
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 11))

    lon_min, lon_max, lat_min, lat_max = bounds

    # Draw country polygons
    draw_countries(ax, countries,
                   lon_range=(lon_min - 5, lon_max + 5),
                   lat_range=(lat_min - 5, lat_max + 5),
                   fill_color='#2a2a2a',
                   edge_color='#888888',
                   linewidth=0.7)

    # Draw water (background)
    ax.set_facecolor('#0a1522')  # dark blue for sea

    # Draw beam paths from each station to the target
    tgt_name, tgt_lat, tgt_lon = target
    for stn_name, stn_lat, stn_lon, stn_desig in stations:
        lats, lons = great_circle_points(stn_lat, stn_lon, tgt_lat, tgt_lon, n=100)
        ax.plot(lons, lats, color=line_color, linewidth=2.5,
                alpha=0.85, zorder=5, linestyle='-')

        # Distance label at midpoint
        mid_idx = len(lats) // 2
        # Compute great circle distance
        R = 6371.0
        lat1r, lon1r = math.radians(stn_lat), math.radians(stn_lon)
        lat2r, lon2r = math.radians(tgt_lat), math.radians(tgt_lon)
        d_km = 2 * R * math.asin(math.sqrt(
            math.sin((lat2r - lat1r) / 2) ** 2 +
            math.cos(lat1r) * math.cos(lat2r) * math.sin((lon2r - lon1r) / 2) ** 2
        ))
        ax.text(lons[mid_idx], lats[mid_idx] + 0.15,
                f'{d_km:.0f} km',
                color=line_color, fontsize=11, fontweight='bold',
                ha='center', va='bottom', zorder=7,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#1a1a1a', edgecolor=line_color,
                          linewidth=1))

    # Draw stations
    for stn_name, stn_lat, stn_lon, stn_desig in stations:
        ax.plot(stn_lon, stn_lat, marker='^', color=station_color,
                markersize=18, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)
        # Station label
        ax.annotate(f'  {stn_name}\n  ({stn_desig})',
                    xy=(stn_lon, stn_lat),
                    xytext=(stn_lon + 0.2, stn_lat + 0.15),
                    color='white', fontsize=11, fontweight='bold',
                    zorder=11,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='#1a1a1a',
                              edgecolor=station_color,
                              linewidth=1.5,
                              alpha=0.9))

    # Draw target
    ax.plot(tgt_lon, tgt_lat, marker='*', color=target_color,
            markersize=28, markeredgecolor='white',
            markeredgewidth=1.5, zorder=10)
    ax.annotate(f'  {tgt_name}',
                xy=(tgt_lon, tgt_lat),
                xytext=(tgt_lon + 0.2, tgt_lat - 0.3),
                color='white', fontsize=12, fontweight='bold',
                zorder=11,
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#1a1a1a',
                          edgecolor=target_color,
                          linewidth=1.5,
                          alpha=0.9))

    # Set bounds
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_aspect('equal')

    # Axis labels
    ax.set_xlabel('Longitude (deg)', fontsize=11)
    ax.set_ylabel('Latitude (deg)', fontsize=11)

    # Grid
    ax.grid(alpha=0.15, color='gray', linestyle=':')

    # Title
    ax.set_title(f'{title}\n{subtitle}',
                 fontsize=14, fontweight='bold', pad=12)

    fig.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#0a1522')
    plt.close(fig)
    print(f"Saved: {filename}")


# ================================================================
#  KNICKEBEIN: Stollberg + Kleve -> Derby
# ================================================================
make_system_map(
    title="Knickebein (31.5 MHz) - Operational Station Pair",
    subtitle="Kleve (Kn-4) and Stollberg (Kn-2) beams intersecting over Derby\nRolls-Royce Merlin engine works target, 21-22 June 1940 beam detection",
    stations=[
        ("Kleve-Materborn",  51.7886, 6.1031,  "Kn-4, 111 m ASL"),
        ("Stollberg",        54.6436, 8.9447,  "Kn-2, 72 m ASL"),
    ],
    target=("Derby (Rolls-Royce)", 52.9220, -1.4750),
    bounds=(-3, 11, 50.5, 56),
    filename="/home/alan/claude/BotB/map_knickebein.png",
)

# ================================================================
#  X-GERÄT: Cotentin + Audembert -> Coventry
# ================================================================
make_system_map(
    title="X-Gerät (66-75 MHz) - Operational Station Pair",
    subtitle="Weser (Cotentin director) and Elbe/Oder/Rhein (Audembert cross beams)\nCoventry 'Moonlight Sonata' raid, 14/15 November 1940",
    stations=[
        ("Weser/Spree (Cotentin)",  49.6898, -1.9326, "Director, 170 m ASL"),
        ("Elbe/Oder/Rhein (Audembert)", 50.8614, 1.6931,  "Cross beams, 84 m ASL"),
    ],
    target=("Coventry", 52.4081, -1.5106),
    bounds=(-5, 5, 49, 54),
    filename="/home/alan/claude/BotB/map_xgerat.png",
)

# ================================================================
#  Y-GERÄT: Cassel + Beaumont-Hague -> Liverpool
# ================================================================
make_system_map(
    title="Y-Gerät / Wotan II / Benito (42-48 MHz) - Primary Station Pair",
    subtitle="Cassel (Mont Cassel downlink 42.5 MHz) and Beaumont-Hague (Hague peninsula)\nLiverpool/Birkenhead raid, 3/4 May 1941 - three III/KG 26 Heinkels shot down",
    stations=[
        ("Cassel (Mont Cassel)",  50.8006, 2.4883,  "Wotan II, 196 m ASL"),
        ("Beaumont-Hague",        49.6733, -1.8525, "Hague peninsula, 197 m ASL"),
    ],
    target=("Liverpool/Birkenhead", 53.4084, -2.9916),
    bounds=(-5, 5, 49, 55),
    filename="/home/alan/claude/BotB/map_ygerat.png",
)

print("\nAll maps generated.")
