"""
Knickebein 31.5 MHz beam map: Kleve and Stollberg transmitters with
beams to Derby. The Kleve beam passes over Spalding on its way; the
Stollberg beam passes over Beeston on its way. Both equisignals cross
at Derby (Rolls-Royce Merlin works) and extend past it. An He 111 H-3
silhouette flies the Kleve->Derby equisignal -- the only Knickebein
path that survives the spherical-Earth Fock-diffraction null hypothesis.
"""
import json
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch as MplPatch
from matplotlib.lines import Line2D

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)       # for make_p526_vs_p368_graphs (sibling in Knickebein/)
sys.path.insert(0, _REPO_ROOT)  # for botb_itu_analysis + grwave at repo root
from make_p526_vs_p368_graphs import sn_snr_peak, p526_snr_peak  # noqa: E402
from botb_itu_analysis import CROSSOVER_dB, N_FLOOR_dBW  # noqa: E402

# SNR → V_rx(μV) at 50 Ω: V_rx_dBμV = SNR + (N_FLOOR_dBW + 137).
# Computed dynamically so any bandwidth change in botb_itu_analysis.py
# propagates here without manual edits.
SNR_TO_DBUV = N_FLOOR_dBW + 137.0
NOISE_UV = 10 ** (SNR_TO_DBUV / 20.0)  # noise floor at +0 dB SNR

# ---------------------------------------------------------------- projection
LAT_REF = 53.0
COS_LAT = math.cos(math.radians(LAT_REF))


def to_uv(lat, lon):
    """(lat, lon) -> equal-aspect projected (u, v)."""
    return lon * COS_LAT, lat


def from_uv(u, v):
    return v, u / COS_LAT


def project_point_onto_line(p, a, b):
    """Project lat/lon point p onto line through a,b in projected space."""
    pu, pv = to_uv(*p)
    au, av = to_uv(*a)
    bu, bv = to_uv(*b)
    du, dv = bu - au, bv - av
    t = ((pu - au) * du + (pv - av) * dv) / (du * du + dv * dv)
    return from_uv(au + t * du, av + t * dv)


def beam_geometry(tx, rx, half_deg, extend_factor):
    """Return (centre_end, dot_end, dash_end) lat/lon endpoints of a
    straight-line beam in projected space."""
    txu, txv = to_uv(*tx)
    rxu, rxv = to_uv(*rx)
    du, dv = rxu - txu, rxv - txv
    length = math.hypot(du, dv) * extend_factor
    angle = math.atan2(dv, du)
    h = math.radians(half_deg)
    end_c = (txu + length * math.cos(angle),
             txv + length * math.sin(angle))
    end_d = (txu + length * math.cos(angle - h),
             txv + length * math.sin(angle - h))
    end_s = (txu + length * math.cos(angle + h),
             txv + length * math.sin(angle + h))
    return (from_uv(*end_c), from_uv(*end_d), from_uv(*end_s))


# ---------------------------------------------------------------- stations
KLEVE = (51.7886, 6.1031)
STOLLBERG = (54.6436, 8.9447)
DERBY = (52.9220, -1.4750)
SPALDING_RAW = (52.7870, -0.1530)
BEESTON_RAW = (52.9270, -1.2150)
SPALDING = project_point_onto_line(SPALDING_RAW, KLEVE, DERBY)
BEESTON = project_point_onto_line(BEESTON_RAW, STOLLBERG, DERBY)

SQUINT_HALF_DEG = 2.5
EXAGGERATE = 1.6
DRAW_HALF = SQUINT_HALF_DEG * EXAGGERATE
EXTEND = 1.20


def gc_distance_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    return 2 * 6371.0 * math.asin(math.sqrt(
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


D_KLEVE_SPALDING = gc_distance_km(KLEVE, SPALDING)
D_KLEVE_DERBY = gc_distance_km(KLEVE, DERBY)
D_STOLLBERG_BEESTON = gc_distance_km(STOLLBERG, BEESTON)
D_STOLLBERG_DERBY = gc_distance_km(STOLLBERG, DERBY)


def snr_to_uv(snr_db):
    return 10 ** ((snr_db + SNR_TO_DBUV) / 20.0)


# (label, distance_km, tx_height_m, rx_height_m, ground)
PRED_PATHS = [
    ("Kleve\u2192Spalding",     D_KLEVE_SPALDING,    111, 6000, "land"),
    ("Kleve\u2192Derby",        D_KLEVE_DERBY,       111, 6000, "land"),
    ("Stollberg\u2192Beeston",  D_STOLLBERG_BEESTON,  72, 6000, "sea"),
    ("Stollberg\u2192Derby",    D_STOLLBERG_DERBY,    72, 6000, "sea"),
]

PREDICTIONS = []
for name, d, tx, rx, g in PRED_PATHS:
    # Apply the 5° squint equisignal crossover loss (CROSSOVER_dB = -19.0
    # dB) so the values plotted on the map match the legend "Equisignal
    # Strength in μV" and the null-doc table. The peak SNR functions
    # return sub-beam peak; the pilot tracks the equisignal, which is
    # 19 dB below either peak for the Telefunken Large Knickebein.
    sn_uv   = snr_to_uv(sn_snr_peak(d, tx, rx, ground_name=g) + CROSSOVER_dB)
    fock_uv = snr_to_uv(p526_snr_peak(d, tx, rx, ground_name=g) + CROSSOVER_dB)
    PREDICTIONS.append({"name": name, "d": d, "sn": sn_uv, "fock": fock_uv})

# ---------------------------------------------------------------- coastlines
GEOJSON = "/tmp/countries.geojson"
with open(GEOJSON) as f:
    COUNTRIES = json.load(f)


def draw_countries(ax, lon_min, lon_max, lat_min, lat_max,
                   facecolor="#1f2638", edgecolor="#5a6378", linewidth=0.6):
    for feat in COUNTRIES["features"]:
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            outer = np.array(poly[0])
            if (outer[:, 0].max() < lon_min - 2
                    or outer[:, 0].min() > lon_max + 2
                    or outer[:, 1].max() < lat_min - 2
                    or outer[:, 1].min() > lat_max + 2):
                continue
            ax.add_patch(MplPolygon(outer, closed=True,
                                    facecolor=facecolor, edgecolor=edgecolor,
                                    linewidth=linewidth, zorder=1))


# ---------------------------------------------------------------- beam draw
def draw_beam(ax, tx, rx, status, label_extra="", extend=None):
    if extend is None:
        extend = EXTEND
    end_c, end_d, end_s = beam_geometry(tx, rx, DRAW_HALF, extend)

    if status == "operational":
        dot_c, dash_c, eq_c = "#C040E0", "#3070E0", "#FFE640"
        a_fill, eq_lw, eq_ls = 0.32, 2.6, "-"
    else:
        dot_c, dash_c, eq_c = "#6A3580", "#2F4870", "#A89020"
        a_fill, eq_lw, eq_ls = 0.18, 1.6, (0, (5, 4))

    # cone polygons (TX -> dot edge -> centreline -> TX)
    dot_lon = [tx[1], end_d[1], end_c[1]]
    dot_lat = [tx[0], end_d[0], end_c[0]]
    dash_lon = [tx[1], end_c[1], end_s[1]]
    dash_lat = [tx[0], end_c[0], end_s[0]]

    ax.fill(dot_lon, dot_lat, facecolor=dot_c, alpha=a_fill,
            edgecolor=dot_c, linewidth=0.5, zorder=3)
    ax.fill(dash_lon, dash_lat, facecolor=dash_c, alpha=a_fill,
            edgecolor=dash_c, linewidth=0.5, zorder=3)
    # centreline
    ax.plot([tx[1], end_c[1]], [tx[0], end_c[0]],
            color=eq_c, linewidth=eq_lw, linestyle=eq_ls,
            zorder=4, solid_capstyle="round")
    return end_c


# ---------------------------------------------------------------- He 111 H-3
def draw_he111(ax, lat, lon, bearing_deg, scale=1.0):
    """
    Top-down silhouette of an He 111 H-3.
    Wingspan normalised to 1.0; total length 0.73 (real ratio 22.6 m / 16.4 m).
    Distinctive features: bulbous glazed nose, twin Jumo 211 nacelles set
    inboard, low-aspect wing with slightly tapered planform, single fin.
    """
    # half wing planform (tip at x=0.50, root at x=0.045)
    wing = np.array([
        [0.045,  0.10],   # root LE
        [0.20,   0.13],   # past inner engine
        [0.40,   0.06],
        [0.50,   0.00],
        [0.40,  -0.06],
        [0.20,  -0.07],
        [0.045, -0.06],
        [-0.045, -0.06],
        [-0.20,  -0.07],
        [-0.40,  -0.06],
        [-0.50,   0.00],
        [-0.40,   0.06],
        [-0.20,   0.13],
        [-0.045,  0.10],
    ])
    # fuselage with bulbous glazed nose
    body = np.array([
        [0.000,  0.430],
        [0.038,  0.395],
        [0.052,  0.350],
        [0.052,  0.250],
        [0.046,  0.100],
        [0.046, -0.080],
        [0.038, -0.220],
        [0.026, -0.290],
        [0.012, -0.305],
        [-0.012, -0.305],
        [-0.026, -0.290],
        [-0.038, -0.220],
        [-0.046, -0.080],
        [-0.046,  0.100],
        [-0.052,  0.250],
        [-0.052,  0.350],
        [-0.038,  0.395],
        [0.000,   0.430],
    ])
    # twin Jumo 211 nacelles, extending forward of wing LE
    nacelle_r = np.array([
        [0.20,  0.27],   # spinner tip
        [0.235, 0.24],
        [0.235, 0.05],
        [0.21,  -0.05],
        [0.165, -0.05],
        [0.165,  0.05],
        [0.18,   0.24],
    ])
    nacelle_l = nacelle_r.copy()
    nacelle_l[:, 0] *= -1
    # propeller hubs (small dark dots)
    prop_r_hub = (0.20, 0.27)
    prop_l_hub = (-0.20, 0.27)
    # propeller blur disks
    prop_r_blur = np.array([
        [0.20,  0.30], [0.215, 0.28], [0.215, 0.26],
        [0.20,  0.24], [0.185, 0.26], [0.185, 0.28],
    ])
    prop_l_blur = prop_r_blur.copy()
    prop_l_blur[:, 0] *= -1
    # horizontal tail
    tail_h = np.array([
        [0.030, -0.250],
        [0.140, -0.275],
        [0.140, -0.305],
        [0.030, -0.290],
        [-0.030, -0.290],
        [-0.140, -0.305],
        [-0.140, -0.275],
        [-0.030, -0.250],
    ])
    # vertical fin (top-down: thin sliver along centreline)
    tail_v = np.array([
        [0.010, -0.230],
        [0.014, -0.305],
        [-0.014, -0.305],
        [-0.010, -0.230],
    ])
    # glazed nose (light blue cap)
    glaze = np.array([
        [0.000, 0.430],
        [0.030, 0.405],
        [0.045, 0.370],
        [0.045, 0.300],
        [-0.045, 0.300],
        [-0.045, 0.370],
        [-0.030, 0.405],
    ])
    # ventral gondola (bola) shadow
    gondola = np.array([
        [0.025, 0.12],
        [0.030, -0.05],
        [0.020, -0.18],
        [-0.020, -0.18],
        [-0.030, -0.05],
        [-0.025, 0.12],
    ])
    # Balkenkreuz markings (German cross) -- just two thin bars on each wing
    cross_r = np.array([
        [0.34,  0.025], [0.36, 0.025],
        [0.36, -0.025], [0.34, -0.025],
    ])
    cross_l = cross_r.copy()
    cross_l[:, 0] *= -1

    angle = math.radians(-bearing_deg)
    cs, sn = math.cos(angle), math.sin(angle)
    Rmat = np.array([[cs, -sn], [sn, cs]])
    cos_lat = math.cos(math.radians(lat))

    def to_map(pts):
        local = pts @ Rmat.T
        local[:, 0] = lon + (local[:, 0] * scale) / cos_lat
        local[:, 1] = lat + local[:, 1] * scale
        return local

    # Luftwaffe night-bomber paint:
    #   RLM 71 Dunkelgrün  -> #4A5934   (mid-tone wing/fuselage)
    #   RLM 70 Schwarzgrün -> #2A361E   (darker shading on engines / fin)
    #   RLM 22 Schwarz     -> #181818   (engines, undercoat)
    #   Balkenkreuz        -> black, with white outline
    layers = [
        (wing,      "#4A5934", "#13180D", 0.8, 14),
        (gondola,   "#2A361E", "#13180D", 0.6, 14),
        (tail_h,    "#4A5934", "#13180D", 0.8, 14),
        (nacelle_l, "#1F1F1F", "#0a0a0a", 0.7, 15),
        (nacelle_r, "#1F1F1F", "#0a0a0a", 0.7, 15),
        (body,      "#536440", "#13180D", 0.9, 16),
        (tail_v,    "#2A361E", "#13180D", 0.6, 17),
        (glaze,     "#A6C8D6", "#13180D", 0.6, 18),
        (cross_l,   "#0a0a0a", "#FFFFFF", 0.6, 19),
        (cross_r,   "#0a0a0a", "#FFFFFF", 0.6, 19),
        (prop_l_blur, "#1a1a1a", "#1a1a1a", 0.0, 19),
        (prop_r_blur, "#1a1a1a", "#1a1a1a", 0.0, 19),
    ]
    for pts, fc, ec, lw, z in layers:
        ax.add_patch(MplPolygon(to_map(pts), closed=True,
                                facecolor=fc, edgecolor=ec,
                                linewidth=lw, alpha=0.95, zorder=z))
    # propeller hubs
    for hub in (prop_l_hub, prop_r_hub):
        local = (np.array([hub]) @ Rmat.T)[0]
        x = lon + (local[0] * scale) / cos_lat
        y = lat + local[1] * scale
        ax.plot(x, y, "o", color="#1a1a1a", markersize=2.5, zorder=20)


# ---------------------------------------------------------------- figure
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(15.5, 12))
fig.patch.set_facecolor("#080d18")
ax.set_facecolor("#0a1626")

bounds = (-6.0, 12.5, 49.8, 56.8)
draw_countries(ax, *bounds)

# only two beams now: each TX -> Derby (passing through the intermediate point)
beams_to_draw = [
    (KLEVE,     DERBY, "operational"),   # passes through SPALDING
    (STOLLBERG, DERBY, "phantom"),       # passes through BEESTON
]


# ---- SN / Fock strength wedges along the Kleve→Derby centreline -------
# Green = Sommerfeld-Norton flat-Earth (stays USABLE the whole way).
# Pink  = P.526 Fock globe diffraction: solid until the equisignal
#         crosses the noise floor, hatched from that crossover onward
#         (UNUSABLE). Ported from the Telefunken variant; scales linearly
#         along the Kleve beam (no VISUAL_FRACS piecewise).
def _fock_eq_vrx_uv(d_km, tx_m, rx_m, ground):
    snr_peak = p526_snr_peak(d_km, tx_m=tx_m, rx_m=rx_m, ground_name=ground)
    return NOISE_UV * 10 ** ((snr_peak + CROSSOVER_dB) / 20.0)


def _find_fock_crossover_km(tx_m, rx_m, ground, d_lo=100, d_hi=1200, step=2):
    for d in range(d_lo, d_hi + 1, step):
        if _fock_eq_vrx_uv(d, tx_m, rx_m, ground) < NOISE_UV:
            return d
    return d_hi


def draw_fe_ge_wedges(ax, tx, rx, distance_km, beam_end_km, crossover_km,
                       half_deg=0.9):
    """Green SN + pink Fock wedges hugging the yellow equisignal centreline.
    Green extends to beam_end_km; pink goes solid to crossover_km and
    hatched from there to beam_end_km."""
    txu, txv = to_uv(*tx)
    rxu, rxv = to_uv(*rx)
    du, dv = rxu - txu, rxv - txv
    length_full = math.hypot(du, dv)      # projected length of TX→RX (= distance_km)
    angle_c = math.atan2(dv, du)
    h_rad = math.radians(half_deg)

    def edge_pt(d_km, sign):
        L = length_full * (d_km / distance_km)
        a = angle_c + sign * h_rad
        return from_uv(txu + L * math.cos(a), txv + L * math.sin(a))

    def los_pt(d_km):
        frac = d_km / distance_km
        return from_uv(txu + frac * du, txv + frac * dv)

    # Green FE wedge: TX → edge at beam_end (dot side) → LoS at beam_end → TX
    g_edge = edge_pt(beam_end_km, -1)
    g_tip = los_pt(beam_end_km)
    ax.fill([tx[1], g_edge[1], g_tip[1]],
            [tx[0], g_edge[0], g_tip[0]],
            facecolor="#00E676", alpha=0.55,
            edgecolor="#00E676", linewidth=0.6, zorder=3.4)

    # Pink Fock solid wedge: TX → edge at crossover (dash side) → LoS at crossover → TX
    co = min(crossover_km, beam_end_km)
    p_edge_x = edge_pt(co, +1)
    p_tip_x = los_pt(co)
    ax.fill([tx[1], p_edge_x[1], p_tip_x[1]],
            [tx[0], p_edge_x[0], p_tip_x[0]],
            facecolor="#FF3399", alpha=0.55,
            edgecolor="#FF3399", linewidth=0.6, zorder=3.4)

    # Pink Fock hatched trapezoid: crossover → beam end (UNUSABLE past here)
    if co < beam_end_km:
        p_edge_end = edge_pt(beam_end_km, +1)
        p_tip_end = los_pt(beam_end_km)
        ax.add_patch(MplPolygon(
            [(p_tip_x[1], p_tip_x[0]),
             (p_edge_x[1], p_edge_x[0]),
             (p_edge_end[1], p_edge_end[0]),
             (p_tip_end[1], p_tip_end[0])],
            closed=True, facecolor="none", edgecolor="#FF3399",
            linewidth=0.5, hatch="///", alpha=0.55, zorder=3.4))


KLEVE_FOCK_CROSSOVER_KM = _find_fock_crossover_km(111, 6000, "land")
STOLL_FOCK_CROSSOVER_KM = _find_fock_crossover_km(72, 6000, "sea")

# Both yellow beams span 1000 km physical so the reader can compare
# Kleve vs Stollberg reach at a glance. Per-beam extend = 1000 / TX→Derby km.
BEAM_VISUAL_KM = 1000

plane_done = False
for tx, rx, status in beams_to_draw:
    distance_km = gc_distance_km(tx, rx)
    beam_extend = BEAM_VISUAL_KM / distance_km
    end_c = draw_beam(ax, tx, rx, status, extend=beam_extend)
    # Wedges run along the equisignal centreline to the beam tip (1000 km).
    # Kleve uses 111 m / land; Stollberg uses 72 m / sea.
    if tx is KLEVE and status == "operational":
        draw_fe_ge_wedges(ax, tx, rx, distance_km,
                          beam_end_km=BEAM_VISUAL_KM,
                          crossover_km=KLEVE_FOCK_CROSSOVER_KM)
    elif tx is STOLLBERG:
        draw_fe_ge_wedges(ax, tx, rx, distance_km,
                          beam_end_km=BEAM_VISUAL_KM,
                          crossover_km=STOLL_FOCK_CROSSOVER_KM)
    if not plane_done and tx is KLEVE:
        # fly the equisignal over the North Sea, between Kleve and the
        # English coast -- well clear of the Derby/Spalding label cluster.
        t = 0.40
        plane_lat = tx[0] + t * (DERBY[0] - tx[0])
        plane_lon = tx[1] + t * (DERBY[1] - tx[1])
        du = (DERBY[1] - tx[1]) * COS_LAT
        dv = DERBY[0] - tx[0]
        bearing_from_north = math.degrees(math.atan2(du, dv))
        draw_he111(ax, plane_lat, plane_lon, bearing_from_north, scale=1.20)
        ax.annotate("He 111 H-3\nriding the Kleve\u2192Derby equisignal\n"
                    "altitude 6 000 m",
                    xy=(plane_lon + 0.40, plane_lat + 0.20),
                    xytext=(plane_lon + 1.6, plane_lat + 0.95),
                    color="white", fontsize=10, fontweight="bold", zorder=22,
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#181828",
                              edgecolor="#80C8FF", linewidth=1.2, alpha=0.94))
        plane_done = True

# transmitters
for lat, lon, name, desig in [
    (KLEVE[0], KLEVE[1], "Kleve",     "Kn-4"),
    (STOLLBERG[0], STOLLBERG[1], "Stollberg", "Kn-2"),
]:
    ax.plot(lon, lat, marker="^", markersize=22, color="#FF4040",
            markeredgecolor="white", markeredgewidth=1.6, zorder=20)
    ax.plot([lon, lon], [lat, lat + 0.18], color="#FF4040",
            linewidth=1.2, zorder=19)
    ax.annotate(f"  {name}  ({desig})\n  31.5 MHz, 3 kW",
                xy=(lon, lat), xytext=(lon + 0.3, lat + 0.30),
                color="white", fontsize=10.5, fontweight="bold", zorder=21,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#13131f",
                          edgecolor="#FF4040", linewidth=1.3, alpha=0.92))

# targets (Spalding / Beeston now snapped onto the beams)
target_data = [
    (SPALDING[0], SPALDING[1], "Spalding",
     f"{D_KLEVE_SPALDING:.0f} km from Kleve",
     (-0.85, 0.95, "left")),
    (BEESTON[0],  BEESTON[1],  "Beeston",
     f"{D_STOLLBERG_BEESTON:.0f} km from Stollberg",
     (-1.5, 0.6, "right")),
    (DERBY[0],    DERBY[1],    "Derby",
     f"{D_KLEVE_DERBY:.0f} km Kleve  \u2502  "
     f"{D_STOLLBERG_DERBY:.0f} km Stollberg",
     (0.0, -0.90, "center")),
]
for lat, lon, name, note, (dx, dy, ha) in target_data:
    ax.plot(lon, lat, marker="o", markersize=18, color="none",
            markeredgecolor="#80E0FF", markeredgewidth=2.0, zorder=20)
    ax.plot(lon, lat, marker="o", markersize=8, color="#80E0FF", zorder=21)
    ax.annotate(f"{name}\n{note}",
                xy=(lon, lat), xytext=(lon + dx, lat + dy),
                color="white", fontsize=10, ha=ha, zorder=21,
                arrowprops=dict(arrowstyle="-", color="#80E0FF", lw=0.8,
                                alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#13131f",
                          edgecolor="#80E0FF", linewidth=1.1, alpha=0.92))

# range rings (200 km steps) around each TX
for tx in (KLEVE, STOLLBERG):
    txu, txv = to_uv(*tx)
    for d_km in (200, 400, 600, 800):
        # convert km -> degrees of latitude (~111 km/deg) and project radius
        # in (u, v) it's circle of radius d_km / 111.0 (degrees)
        r_deg = d_km / 111.0
        ang = np.linspace(0, 2 * np.pi, 180)
        u = txu + r_deg * np.cos(ang)
        v = txv + r_deg * np.sin(ang)
        ax.plot(u / COS_LAT, v, color="#3a4860", linewidth=0.6,
                linestyle=":", alpha=0.55, zorder=2)

# bounds + cosmetic
ax.set_xlim(bounds[0], bounds[1])
ax.set_ylim(bounds[2], bounds[3])
ax.set_aspect(1.0 / COS_LAT)
ax.set_xlabel("Longitude  \u00b0E", color="white", fontsize=10)
ax.set_ylabel("Latitude  \u00b0N", color="white", fontsize=10)
ax.grid(alpha=0.10, color="gray", linestyle=":")
ax.tick_params(colors="white")

ax.set_title("Knickebein 31.5 MHz \u2014 Kleve & Stollberg beams crossing over Derby\n"
             "He 111 H-3 riding the Kleve\u2192Derby equisignal (via Spalding)",
             fontsize=12.5, fontweight="bold", color="white", pad=14)

# --- Top section ------------------------------------------------------
knickebein_section = [
    ("Knickebein 31.5 MHz, 3 kW", "#FFFFFF", True),
    ("  DOT  (purple)",           "#C040E0", False),
    ("  DASH (blue)",             "#3070E0", False),
    ("  EQUI (yellow)",           "#FFE640", False),
]
# --- Bottom section: full-width Signal Strength ---------------------
# Target distances live inside the inset bar-chart labels (e.g.
# "Kleve→Spalding 440 km"), so the standalone "Equisignal Beam Dist to
# Target" block is redundant and has been removed for the compact layout.
signal_section = [
    ("Equisignal Strength in \u00b5V", "#FFFFFF", True),
    ("  \u25A0  Sommerfeld-Norton FE", "#3CD46A", False),
    ("  \u25A0  P.526 Fock (globe)", "#FF3399", False),
    (f"  \u2500  noise floor  {NOISE_UV:.2f} \u00b5V", "#FFFFFF", False),
]

legend_x       = 0.012   # top-left corner of the map
line_h         = 0.024
blank_h        = line_h * 0.40
legend_w       = 0.150
LEG_FONTSIZE   = 9.5

# Heights (in axes fraction)
top_section_rows  = len(knickebein_section)
top_section_h     = top_section_rows * line_h
sig_section_h     = len(signal_section) * line_h
legend_block_h    = top_section_h + blank_h + sig_section_h + 0.020

# Anchor legend block to the top-left of the map.
legend_top_y = 0.997
legend_bot_y = legend_top_y - legend_block_h
ax.add_patch(plt.Rectangle((legend_x - 0.005, legend_bot_y - 0.005),
                           legend_w, legend_block_h + 0.005,
                           transform=ax.transAxes,
                           facecolor="#0a0e1c", edgecolor="#3a4258",
                           linewidth=1.0, alpha=0.94, zorder=24))


def _draw_row(x, y, text, color, bold):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=LEG_FONTSIZE, color=color or "#D6D6E0",
            fontweight="bold" if bold else "normal",
            family="monospace", va="top", ha="left", zorder=25)


# --- top: Knickebein column ---
y0 = legend_top_y - 0.010
for i, (text, color, bold) in enumerate(knickebein_section):
    _draw_row(legend_x, y0 - i * line_h, text, color, bold)

# --- bottom: Signal Strength ---
y0 = legend_top_y - top_section_h - blank_h - 0.010
for i, (text, color, bold) in enumerate(signal_section):
    _draw_row(legend_x, y0 - i * line_h, text, color, bold)

# ---------------------------------------------------------------- inset
# Mini horizontal bar chart of predicted field strength at each target.
# Lives in the empty Belgium / Netherlands / Channel area in the lower
# centre-right of the map.  Two bars per path: Sommerfeld-Norton flat
# Earth (what you'd get without curvature) and ITU-R P.526 Fock globe
# diffraction (what physics actually says you get).  Vertical guide
# line at the +0 dB receiver noise floor (0.195 μV into 50 Ω).
inset = ax.inset_axes([0.012, 0.025, 0.358, 0.270])
inset.set_xscale("log")
# Match the Telefunken variant's x-range so the UNUSABLE text at the
# left edge sits inside the chart box (wider log spans visually compress
# the 0.02..0.1 μV band off-screen).
inset.set_xlim(0.02, 3e3)
inset.set_facecolor("#0a0e1c")
for s in inset.spines.values():
    s.set_edgecolor("#3a4258")

# Order by descending distance so the hardest path (weakest signal) sits
# at the top of the chart, matching the Telefunken variant's layout.
PREDICTIONS_SORTED = sorted(PREDICTIONS, key=lambda p: p["d"], reverse=True)

n = len(PREDICTIONS_SORTED)
bar_h = 0.20
group_h = 0.72
y_centres = []
for i, p in enumerate(PREDICTIONS_SORTED):
    y0 = (n - 1 - i) * group_h
    # Path label lives INSIDE the chart, just above this group's green bar.
    # Colour encodes the TX: Kleve paths bright yellow, Stollberg amber.
    label_color = "#FFE640" if p["name"].startswith("Kleve") else "#A89020"
    inset.text(0.025, y0 + bar_h/2 + 0.04 + 0.20,
               f"{p['name']}  {p['d']:.0f} km",
               fontsize=9.0, color=label_color, ha="left", va="center",
               fontweight="bold", zorder=30)
    inset.barh(y0 + bar_h/2 + 0.04, p["sn"],   height=bar_h,
               color="#3CD46A", edgecolor="#3CD46A", alpha=0.85,
               label="Sommerfeld-Norton FE" if i == 0 else None)
    inset.barh(y0 - bar_h/2 - 0.04, p["fock"], height=bar_h,
               color="#FF3399", edgecolor="#FF3399", alpha=0.85,
               label="P.526 Fock (globe)" if i == 0 else None)
    y_centres.append(y0)
    if p["sn"] >= 1000:
        sn_lbl = f"{p['sn']:,.0f} \u00b5V"
    else:
        sn_lbl = f"{p['sn']:.0f} \u00b5V"
    # Decimal notation so sub-μV values read as 0.020 / 0.012 rather than
    # scientific notation.
    if p["fock"] >= 10:
        fk_lbl = f"{p['fock']:.1f} \u00b5V"
    elif p["fock"] >= 1:
        fk_lbl = f"{p['fock']:.2f} \u00b5V"
    elif p["fock"] > 0:
        _places = max(3, -int(math.floor(math.log10(p["fock"]))) + 1)
        fk_lbl = f"{p['fock']:.{_places}f} \u00b5V"
    else:
        fk_lbl = "0 \u00b5V"
    # Green-bar value: inside the bar, right-aligned near the tip, white.
    inset.text(p["sn"] * 0.96, y0 + bar_h/2 + 0.04, sn_lbl,
               fontsize=8.0, color="#FFFFFF", va="center", ha="right",
               fontweight="bold")
    # Pink-bar value: inside the bar if long enough, otherwise to the
    # right of the bar tip. White in both cases.
    if p["fock"] >= 5.0:
        inset.text(p["fock"] * 0.95, y0 - bar_h/2 - 0.04, fk_lbl,
                   fontsize=8.0, color="#FFFFFF", va="center", ha="right",
                   fontweight="bold")
    else:
        inset.text(0.366, y0 - bar_h/2 - 0.04, fk_lbl,
                   fontsize=8.0, color="#FFFFFF", va="center", ha="left",
                   fontweight="bold")
    # Green bar is always USABLE (SN flat-Earth always clears the noise
    # floor at these distances / powers).
    inset.text(0.040, y0 + bar_h/2 + 0.04, "USABLE",
               fontsize=7.0, color="#FFFFFF", va="center", ha="center",
               fontweight="bold")
    # Pink bar: binary USABLE / UNUSABLE relative to the noise floor
    # (matches the Telefunken variant; drops the separate Marginal tier).
    fock_status = "USABLE" if p["fock"] >= NOISE_UV else "UNUSABLE"
    if fock_status == "UNUSABLE":
        inset.barh(y0 - bar_h/2 - 0.04, NOISE_UV, height=bar_h,
                   facecolor="none", edgecolor="#FF3399",
                   linewidth=0.3, hatch="///", alpha=0.45, zorder=2)
    # Hatched extension: from where the solid (or UNUSABLE-hatched) pink
    # visually terminates out to where the green FE bar ends. Makes the
    # Fock-vs-SN signal gap read at a glance.
    pink_end = p["fock"] if fock_status == "USABLE" else NOISE_UV
    if p["sn"] > pink_end:
        inset.barh(y0 - bar_h/2 - 0.04, p["sn"] - pink_end, height=bar_h,
                   left=pink_end, facecolor="none", edgecolor="#FF3399",
                   linewidth=0.3, hatch="///", alpha=0.45, zorder=2)
    inset.text(0.040, y0 - bar_h/2 - 0.04, fock_status,
               fontsize=7.0, color="#FFFFFF", va="center", ha="center",
               fontweight="bold", zorder=5)

inset.set_yticks([])  # labels now live inside the chart above each group
inset.set_ylim(-0.36, (n - 1) * group_h + 0.45)

# noise-floor guide line (legend explains colours)
inset.axvline(NOISE_UV, color="#FFFFFF", linewidth=1.0, alpha=0.85, zorder=4)

inset.tick_params(axis="x", labelsize=7.5, colors="#D6D6E0", pad=1)
inset.tick_params(axis="y", colors="#D6D6E0", length=0, pad=2)

# Inset legend removed - the colour/line key now lives in the main
# legend under the "Signal Strength" section.

# Title lives in the empty black strip above the top bar group,
# inside the plot area so it doesn't force matplotlib to rescale.

plt.tight_layout()
plt.savefig(os.path.join(_HERE, "graphs", "ITU_Calc_knickebein_beam_map.png"), dpi=140,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved ITU_Calc_knickebein_beam_map.png")
