"""
Knickebein 31.5 MHz beam map (Telefunken-distance variant):
Kleve and Stollberg transmitters with five target points laid out along
the Kleve line-of-sight at the Telefunken Sep-1939 over-sea range
intervals (400, 500, 700, 800, 1000 km from Kleve). Same projection,
same TX positions, same map style as the operational variant -- only
the targets and the Equisignal Strength predictions change.
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
from botb_itu_analysis import CROSSOVER_dB  # noqa: E402 — 5° squint equisignal loss

# NOISE_UV and SNR_TO_DBUV are loaded from the CSV below.

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
DERBY = (52.9220, -1.4750)   # used only as the bearing anchor for Kleve LoS

SQUINT_HALF_DEG = 2.5
EXAGGERATE = 1.6
DRAW_HALF = SQUINT_HALF_DEG * EXAGGERATE
KLEVE_REACH_KM = 800         # operational Kleve reach (Telefunken regen)
STOLL_REACH_KM = 1000        # operational Stollberg reach (Telefunken max)


R_EARTH_KM = 6371.0


def gc_distance_km(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    return 2 * R_EARTH_KM * math.asin(math.sqrt(
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))


def gc_bearing_deg(a, b):
    """Initial bearing from a to b, in degrees from N (clockwise)."""
    p1 = math.radians(a[0])
    p2 = math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x))


def gc_destination(origin, bearing_deg, dist_km):
    """Great-circle endpoint at dist_km from origin along bearing."""
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    b = math.radians(bearing_deg)
    delta = dist_km / R_EARTH_KM
    lat2 = math.asin(math.sin(lat1) * math.cos(delta)
                     + math.cos(lat1) * math.sin(delta) * math.cos(b))
    lon2 = lon1 + math.atan2(
        math.sin(b) * math.sin(delta) * math.cos(lat1),
        math.cos(delta) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


D_KLEVE_DERBY = gc_distance_km(KLEVE, DERBY)
D_STOLLBERG_DERBY = gc_distance_km(STOLLBERG, DERBY)

# Pick a Kleve LoS bearing that lands the 1000 km marker on the Irish
# NW shoreline (Donegal Bay) rather than well out in the Atlantic.  This
# is a few degrees north of Kleve->Derby (≈287°), which keeps the beam
# crossing the UK roughly up the spine and dropping into Ireland.
KLEVE_TARGET_BEARING = 296.0

# Telefunken Sep 1939 over-sea range intervals applied to the Kleve LoS.
TF_DISTANCES = [400, 500, 700, 800, 1000]
# Real great-circle distances per Telefunken Sep-1939 range table.
# KL = Kleve→target, ST = Stollberg→target. Nominal "TF X km" label = ST.
KL_DISTANCES   = {400: 392, 500: 450, 700: 607, 800: 696, 1000: 874}
STOLL_DISTANCES = {400: 414, 500: 505, 700: 703, 800: 805, 1000: 1000}


# Visual end of the Kleve beam: both yellow beams (Kleve + Stollberg)
# are held to a consistent 1000 km physical length so the reader can
# compare reach at a glance without per-beam mental conversion.
KLEVE_BEAM_END_KM = 1000
KLEVE_BEAM_END = gc_destination(KLEVE, KLEVE_TARGET_BEARING, KLEVE_BEAM_END_KM)

# Endpoint for the DRAWN yellow Kleve beam: at 95 % along the (u,v) line
# KLEVE→KLEVE_BEAM_END, so the drawn beam sits on the exact same straight
# line that target_along_kleve_los uses for marker placement. Using a
# separate gc_destination(... 950) endpoint would bend slightly off-axis
# due to cylindrical-projection nonlinearity, nudging the blue markers
# off the yellow centreline.
_KBE_FRAC = 0.95
_ku0, _kv0 = to_uv(*KLEVE)
_eu0, _ev0 = to_uv(*KLEVE_BEAM_END)
KLEVE_DRAW_END = from_uv(_ku0 + _KBE_FRAC * (_eu0 - _ku0),
                          _kv0 + _KBE_FRAC * (_ev0 - _kv0))


VISUAL_BEAM_SPAN_KM = 1425   # denominator for visual marker placement;
                              # keeps 1000 km mark near the Irish coast
                              # rather than deep in the interior.

# Per-distance visual fractions along the beam line.  Most use the
# default linear scale; 800 km is nudged closer to 1000 km to sit
# at the NI coast area the user indicated.
VISUAL_FRACS = {
    400: 0.40,
    500: 0.50,
    700: 0.68,
    800: 0.77,
    1000: 0.86,
}


def target_along_kleve_los(d_km):
    """Point at d_km from Kleve along the Kleve->beam_end STRAIGHT LINE
    in projected (u,v) space, scaled so the 1000 km marker lands near the
    coast rather than far along the beam.  Markers sit exactly on the
    drawn yellow equisignal centreline."""
    ku, kv = to_uv(*KLEVE)
    eu, ev = to_uv(*KLEVE_BEAM_END)
    frac = VISUAL_FRACS.get(d_km, d_km / VISUAL_BEAM_SPAN_KM)
    return from_uv(ku + frac * (eu - ku), kv + frac * (ev - kv))


TF_TARGETS = [(d, target_along_kleve_los(d)) for d in TF_DISTANCES]
EXTEND_STOLL = STOLL_REACH_KM / D_STOLLBERG_DERBY


# Load canonical signal strengths from the pre-computed CSV so the
# bar chart values match the master table exactly.
import pandas as pd

_csv = pd.read_csv(os.path.join(_HERE, "botb_signal_strengths.csv"))
# KL-side rows: TF_400km, TF_500km, ... (no ST_ infix). ST-side rows have TF_ST_ prefix.
_tf_kl = _csv[_csv["path_name"].str.match(r"^TF_\d+km$")].copy()
_tf_st = _csv[_csv["path_name"].str.startswith("TF_ST_")].copy()
_tf_st_by_nom = {
    int(r["path_name"].replace("TF_ST_", "").replace("km", "")): r
    for _, r in _tf_st.iterrows()
}

NOISE_UV = _tf_kl["noise_floor_uV"].iloc[0]
SNR_TO_DBUV = 20 * math.log10(NOISE_UV)

PREDICTIONS = []
for _, row in _tf_kl.iterrows():
    # path_name is "TF_400km", "TF_500km", etc. — the "nominal" (Stollberg-side)
    # Telefunken test range. distance_km is the actual Kleve→target Haversine
    # distance, which is shorter (e.g. 392 km for the TF 400 km target). The
    # label on the map is the nominal; the physics row uses the actual Kleve
    # distance; STOLL_DISTANCES keys by the nominal so the Stollberg beam
    # still lands on the same lat/lon target.
    nominal_km = int(row["path_name"].replace("TF_", "").replace("km", ""))
    d_kl = int(row["distance_km"])
    st_row = _tf_st_by_nom[nominal_km]
    PREDICTIONS.append({
        "name": f"TF {nominal_km} km",
        "d": nominal_km,            # used for visual placement (VISUAL_FRACS keys)
        "d_kl": d_kl,               # actual Kleve path length (km)
        "d_st": int(st_row["distance_km"]),  # actual Stollberg path length (km)
        "sn": row["sn_eq_uV"],
        "fock": row["fock_eq_uV"],
        "sn_st": st_row["sn_eq_uV"],
        "fock_st": st_row["fock_eq_uV"],
        "stoll_d": STOLL_DISTANCES[nominal_km],
    })
# Reverse so the bar chart reads 1000 km at top down to 400 km at bottom.
PREDICTIONS.reverse()

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
def draw_beam(ax, tx, rx, status, label_extra="", extend=1.20):
    end_c, end_d, end_s = beam_geometry(tx, rx, DRAW_HALF, extend)

    if status == "operational":
        dot_c, dash_c, eq_c = "#C040E0", "#3070E0", "#FFE640"
        a_fill, eq_lw, eq_ls = 0.32, 2.6, "-"
    else:
        dot_c, dash_c, eq_c = "#6A3580", "#2F4870", "#FFD633"
        a_fill, eq_lw, eq_ls = 0.18, 2.2, (0, (5, 4))

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


def render(HIGHLIGHT_KM):
    # ---------------------------------------------------------------- figure
    plt.style.use("dark_background")
    bounds = (-12.0, 12.5, 49.8, 56.8)
    # Pick a figsize whose aspect equals the data aspect under the
    # 1.0/COS_LAT axes-aspect setting, so the map fills the figure with
    # no empty top/bottom margins.
    _data_w = (bounds[1] - bounds[0]) * COS_LAT
    _data_h = bounds[3] - bounds[2]
    _data_aspect = _data_w / _data_h
    _fig_h = 11.5
    fig, ax = plt.subplots(figsize=(_fig_h * _data_aspect, _fig_h))
    fig.patch.set_facecolor("#080d18")
    ax.set_facecolor("#0a1626")
    draw_countries(ax, *bounds)

    # Both beams aim at Derby; the Kleve beam is extended to cover the
    # 1000 km Telefunken target, the Stollberg beam keeps the original
    # 1.2x visual length.
    # Stollberg beam now aims at the 1000 km Kleve target (where the two
    # beams cross) and extends 25 % past it into the Atlantic.
    TF_1000_POS = TF_TARGETS[-1][1]   # (lat, lon) of the 1000 km marker

    # Stollberg beam aims at whichever target this iteration highlights.
    # Extend factor chosen so the TOTAL Stollberg beam length is the same
    # in every image, equal to the 1000-km-case total (≈12.25 projected
    # units ≈ 1360 visual km).
    HIGHLIGHT_POS = next(pos for d, pos in TF_TARGETS if d == HIGHLIGHT_KM)
    # Stollberg beam aims at the highlighted TF target (TF 1000 km in the
    # 1000-km image, etc.). Length held fixed via STOLL_EXTEND below.
    ST_AIM_POS = HIGHLIGHT_POS
    _su, _sv = to_uv(*STOLLBERG)
    _tu, _tv = to_uv(*ST_AIM_POS)
    _Ls = math.hypot(_tu - _su, _tv - _sv)
    STOLL_TOTAL_LEN = 1000.0 / 111.0   # 1000 km in projected units (≈9.009)
    STOLL_EXTEND = STOLL_TOTAL_LEN / _Ls
    beams_to_draw = [
        (KLEVE,     KLEVE_DRAW_END, "operational", 1.0),
        (STOLLBERG, ST_AIM_POS,     "phantom",     STOLL_EXTEND),
    ]

    # Sommerfeld-Norton flat-Earth vs P.526 Fock globe strength wedges along
    # the Kleve and Stollberg LoSs. Green = flat-Earth (stays USABLE all the
    # way to beam end). Pink = globe diffraction, solid until the equisignal
    # crosses the noise floor, hatched from that crossover to the beam end
    # to signify UNUSABLE.
    def _fock_eq_vrx_uv(d_km, tx_m=72, rx_m=4000, ground="sea"):
        snr_peak = p526_snr_peak(d_km, tx_m=tx_m, rx_m=rx_m, ground_name=ground)
        return NOISE_UV * 10 ** ((snr_peak + CROSSOVER_dB) / 20.0)


    def _find_fock_crossover_km(tx_m=72, rx_m=4000, ground="sea",
                                  d_lo=100, d_hi=1400, step=2):
        for d in range(d_lo, d_hi + 1, step):
            if _fock_eq_vrx_uv(d, tx_m=tx_m, rx_m=rx_m, ground=ground) < NOISE_UV:
                return d
        return d_hi


    def _kleve_visual_frac(d_km):
        """Piecewise-linear visual position along the Kleve LoS, using the
        fiducial VISUAL_FRACS marker positions so arbitrary distances
        (e.g. the Fock crossover) land visually between the expected
        markers rather than at the scrunched d/1425 default."""
        fids = [(0, 0.0)] + sorted(VISUAL_FRACS.items()) \
             + [(1000, 1000 / VISUAL_BEAM_SPAN_KM)]
        if d_km <= 0:
            return 0.0
        if d_km >= 1000:
            return d_km / VISUAL_BEAM_SPAN_KM
        for (d0, f0), (d1, f1) in zip(fids, fids[1:]):
            if d0 <= d_km <= d1:
                t = (d_km - d0) / (d1 - d0)
                return f0 + t * (f1 - f0)
        return d_km / VISUAL_BEAM_SPAN_KM


    def draw_fe_ge_wedges(ax, tx, rx, beam_end_km, crossover_km,
                          half_deg=0.9, use_visual_fracs=True,
                          distance_km=None):
        """Thin green (Fe = SN flat-Earth) and pink (Ge = P.526 Fock globe)
        wedges hugging the yellow equisignal centreline. Green extends
        to beam_end_km; pink is solid to crossover_km then hatched past it.

        Two scaling modes:
          use_visual_fracs=True (default): piecewise via _kleve_visual_frac
            so wedges line up with the on-map TF target markers. Used for
            the Kleve beam where VISUAL_FRACS maps nominal km to scrunched
            visual positions.
          use_visual_fracs=False: pure linear scaling along the TX→rx line.
            Requires distance_km (the physical TX→rx km); beam_end_km and
            crossover_km are then fractions of that distance. Used for the
            Stollberg phantom beam which has no visual-fracs markers."""
        txu, txv = to_uv(*tx)
        eu, ev = to_uv(*rx)
        du, dv = eu - txu, ev - txv
        length_full = math.hypot(du, dv)
        angle_c = math.atan2(dv, du)
        h_rad = math.radians(half_deg)

        if use_visual_fracs:
            end_frac = _kleve_visual_frac(beam_end_km)
            def _scale(d_km):
                return _kleve_visual_frac(d_km) / end_frac
        else:
            if distance_km is None:
                raise ValueError("distance_km required when use_visual_fracs=False")
            def _scale(d_km):
                return d_km / distance_km

        def edge_pt(d_km, sign):
            L = length_full * _scale(d_km)
            a = angle_c + sign * h_rad
            return from_uv(txu + L * math.cos(a), txv + L * math.sin(a))

        def los_pt(d_km):
            frac = _scale(d_km)
            return from_uv(txu + frac * du, txv + frac * dv)

        # Green FE wedge: TX -> edge at beam_end (dot side) -> LoS at beam_end -> TX
        g_edge = edge_pt(beam_end_km, -1)
        g_tip = los_pt(beam_end_km)
        ax.fill([tx[1], g_edge[1], g_tip[1]],
                [tx[0], g_edge[0], g_tip[0]],
                facecolor="#00E676", alpha=0.55,
                edgecolor="#00E676", linewidth=0.6, zorder=3.4)

        # Pink Ge solid wedge: TX -> edge at crossover (dash side) -> LoS at crossover -> TX
        p_edge_x = edge_pt(crossover_km, +1)
        p_tip_x = los_pt(crossover_km)
        ax.fill([tx[1], p_edge_x[1], p_tip_x[1]],
                [tx[0], p_edge_x[0], p_tip_x[0]],
                facecolor="#FF3399", alpha=0.55,
                edgecolor="#FF3399", linewidth=0.6, zorder=3.4)

        # Pink Ge hatched trapezoid: crossover -> beam end (UNUSABLE past here)
        p_edge_end = edge_pt(beam_end_km, +1)
        p_tip_end = los_pt(beam_end_km)
        ax.add_patch(MplPolygon(
            [(p_tip_x[1], p_tip_x[0]),
             (p_edge_x[1], p_edge_x[0]),
             (p_edge_end[1], p_edge_end[0]),
             (p_tip_end[1], p_tip_end[0])],
            closed=True, facecolor="none", edgecolor="#FF3399",
            linewidth=0.5, hatch="///", alpha=0.55, zorder=3.4))


    # Stollberg 72 m TX over sea, RX 4 km — used for both the Kleve wedge
    # (historically rendered with Stollberg-geometry crossover) and the
    # Stollberg phantom-beam wedge added below.
    FOCK_CROSSOVER_KM = _find_fock_crossover_km()

    plane_done = False
    for tx, rx, status, extend in beams_to_draw:
        end_c = draw_beam(ax, tx, rx, status, extend=extend)
        if tx is KLEVE and status == "operational":
            # Wedges extend to the full drawn Kleve beam tip (KLEVE_DRAW_END)
            # using linear scaling along KLEVE→KLEVE_DRAW_END. Crossover km
            # placed so the visual solid→hatched transition sits at the
            # previously-calibrated spot (~1/6 of the TF500↔TF700 gap behind
            # the TF500↔TF700 midpoint = frac 0.56 of Kleve→KLEVE_BEAM_END,
            # i.e. frac 0.56/0.95 of Kleve→KLEVE_DRAW_END).
            kl_d_km = gc_distance_km(KLEVE, KLEVE_DRAW_END)
            KLEVE_SOLID_CROSSOVER_KM = kl_d_km * (0.56 / 0.95)
            draw_fe_ge_wedges(ax, KLEVE, KLEVE_DRAW_END,
                              beam_end_km=kl_d_km,
                              crossover_km=KLEVE_SOLID_CROSSOVER_KM,
                              use_visual_fracs=False,
                              distance_km=kl_d_km)
        if tx is STOLLBERG:
            # Linear scaling along the Stollberg phantom beam (no VISUAL_FRACS
            # markers). distance_km = physical TX→ST_AIM_POS km;
            # beam_end_km = distance_km × STOLL_EXTEND matches the visible
            # beam tip (STOLL_TOTAL_LEN projected units ≈ 1000 km).
            stoll_d_km = gc_distance_km(STOLLBERG, ST_AIM_POS)
            # Visual crossover pinned at an absolute km value (calibrated
            # against the ST→mid(TF500,TF700) aim point, where 0.95 × 736.64
            # ≈ 700 km landed the solid pink just short of the intersection).
            # Keeping it absolute means the solid pink holds the same visual
            # extent from Stollberg no matter where ST_AIM_POS moves.
            STOLL_SOLID_CROSSOVER_KM = 700
            draw_fe_ge_wedges(ax, STOLLBERG, ST_AIM_POS,
                              beam_end_km=stoll_d_km * STOLL_EXTEND,
                              crossover_km=STOLL_SOLID_CROSSOVER_KM,
                              use_visual_fracs=False,
                              distance_km=stoll_d_km)
        if not plane_done and tx is KLEVE:
            # Place plane exactly on the projected yellow centreline.
            _ku, _kv = to_uv(*KLEVE)
            _eu, _ev = to_uv(*KLEVE_BEAM_END)
            _frac = 250 / KLEVE_BEAM_END_KM
            plane_lat, plane_lon = from_uv(
                _ku + _frac * (_eu - _ku), _kv + _frac * (_ev - _kv))
            du = (_eu - _ku)
            dv = (_ev - _kv)
            bearing_from_north = math.degrees(math.atan2(du, dv))
            draw_he111(ax, plane_lat, plane_lon, bearing_from_north, scale=1.20)
            ax.annotate("He 111 H-3\nriding the Kleve\u2192Derby equisignal\n"
                        "altitude 6 000 m",
                        xy=(plane_lon, plane_lat),
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

    # Targets: 5 Telefunken-distance points along the Kleve LoS.
    # Label offsets alternate above/below the beam line so they don't pile up.
    # Pure vertical drop-downs with proportional cascade lengths.
    TARGET_OFFSETS = {
        400:  (0.0, -0.60, "center"),
        500:  (0.0, -0.55, "center"),
        700:  (0.0, -0.9, "center"),
        800:  (0.0, -0.86, "center"),
        1000: (0.0, -0.80, "center"),
    }
    for d_km, (lat, lon) in TF_TARGETS:
        dx, dy, ha = TARGET_OFFSETS[d_km]
        # The 1000 km target is the operational focus of this map variant —
        # give it a brighter cyan glow halo (rings + annotation halo) so it
        # reads distinct from the 400/500/700/800 markers.
        if d_km == HIGHLIGHT_KM:
            for size, alpha in [(48, 0.22), (38, 0.34), (28, 0.48)]:
                ax.plot(lon, lat, marker="o", markersize=size, color="none",
                        markeredgecolor="#80E0FF", markeredgewidth=2.2,
                        alpha=alpha, zorder=19)
        ax.plot(lon, lat, marker="o", markersize=18, color="none",
                markeredgecolor="#80E0FF", markeredgewidth=2.0, zorder=20)
        ax.plot(lon, lat, marker="o", markersize=8, color="#80E0FF", zorder=21)
        if d_km == HIGHLIGHT_KM:
            # Underlaid "halo" annotation: thick, soft cyan arrow and bbox
            # edge drawn behind the real tag to give the line + tag a glow.
            ax.annotate(f"TF {d_km} km\nKl: {KL_DISTANCES[d_km]} | St: {STOLL_DISTANCES[d_km]}",
                        xy=(lon, lat), xytext=(lon + dx, lat + dy),
                        color=(0, 0, 0, 0), fontsize=10, fontweight="bold",
                        ha=ha, zorder=20,
                        arrowprops=dict(arrowstyle="-", color="#80E0FF", lw=5.0,
                                        alpha=0.35),
                        bbox=dict(boxstyle="round,pad=0.55", facecolor="none",
                                  edgecolor="#80E0FF", linewidth=5.0,
                                  alpha=0.30))
            arrow_kw = dict(arrowstyle="-", color="#A5F0FF", lw=1.6, alpha=1.0)
            bbox_kw  = dict(boxstyle="round,pad=0.35", facecolor="#13131f",
                            edgecolor="#A5F0FF", linewidth=2.0, alpha=0.95)
        else:
            arrow_kw = dict(arrowstyle="-", color="#80E0FF", lw=0.8, alpha=0.7)
            bbox_kw  = dict(boxstyle="round,pad=0.35", facecolor="#13131f",
                            edgecolor="#80E0FF", linewidth=1.0, alpha=0.92)
        ax.annotate(f"TF {d_km} km\nKl: {KL_DISTANCES[d_km]} | St: {STOLL_DISTANCES[d_km]}",
                    xy=(lon, lat), xytext=(lon + dx, lat + dy),
                    color="white", fontsize=10, fontweight="bold",
                    ha=ha, zorder=21,
                    arrowprops=arrow_kw,
                    bbox=bbox_kw)

    # range rings removed for cleaner visual

    # bounds + cosmetic
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect(1.0 / COS_LAT)
    ax.set_xlabel("Longitude  \u00b0E", color="white", fontsize=10)
    ax.set_ylabel("Latitude  \u00b0N", color="white", fontsize=10)
    ax.grid(alpha=0.10, color="gray", linestyle=":")
    ax.tick_params(colors="white")

    ax.set_title("Knickebein 31.5 MHz \u2014 Kleve & Stollberg beams "
                 "with Telefunken Sep-1939 range targets along the Kleve LoS\n"
                 "He 111 H-3 riding the Kleve equisignal "
                 "(targets at 400 / 500 / 700 / 800 / 1000 km)",
                 fontsize=12.5, fontweight="bold", color="white", pad=14)

    # --- Top section ------------------------------------------------------
    knickebein_section = [
        ("Knickebein 31.5 MHz, 3 kW", "#FFFFFF", True),
        ("  DOT  (purple)",           "#C040E0", False),
        ("  DASH (blue)",             "#3070E0", False),
        ("  EQUI (yellow)",           "#FFE640", False),
    ]
    # --- Bottom section: full-width Signal Strength ---------------------
    signal_section = [
        ("Equisignal Strength in \u00b5V", "#FFFFFF", True),
        ("  \u25A0  Sommerfeld-Norton (FE)", "#3CD46A", False),
        ("  \u25A0  P.526 Fock (GE)", "#FF3399", False),
        (f"  \u2504  noise floor  {NOISE_UV:.2f} \u00b5V", "#FFFFFF", False),
    ]

    legend_x       = 0.012   # top-left corner of the map
    line_h         = 0.024
    blank_h        = line_h * 0.40
    legend_w       = 0.113
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

    # --- bottom: Signal Strength (with internal 2-column row support) ---
    y0 = legend_top_y - top_section_h - blank_h - 0.010
    for i, entry in enumerate(signal_section):
        y = y0 - i * line_h
        if len(entry) == 3:
            text, color, bold = entry
            _draw_row(legend_x, y, text, color, bold)
        else:
            lt, lc, rt, rc = entry
            _draw_row(legend_x, y, lt, lc, False)
            _draw_row(legend_x + COL2_OFFSET, y, rt, rc, False)

    # ---------------------------------------------------------------- inset
    # Mini horizontal bar chart of predicted field strength at each target.
    # Lives in the empty Belgium / Netherlands / Channel area in the lower
    # centre-right of the map.  Two bars per path: Sommerfeld-Norton flat
    # Earth (what you'd get without curvature) and ITU-R P.526 Fock globe
    # diffraction (what physics actually says you get).  Vertical guide
    # line at the +0 dB receiver noise floor (0.195 μV into 50 Ω).
    inset = ax.inset_axes([0.012, 0.025, 0.300, 0.460])
    inset.set_xscale("log")
    inset.set_xlim(0.02, 3e3)
    inset.set_facecolor("#0a0e1c")
    for s in inset.spines.values():
        s.set_edgecolor("#3a4258")

    n = len(PREDICTIONS)
    bar_h = 0.16
    group_h = 1.10
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle as MplRect

    # Within-group y offsets (relative to group centre y0). ST pair on top,
    # KL pair below. SN↔Fock bars touch within each pair; labels sit in the
    # black space ABOVE their own pair (ST_LABEL above ST pair, KL_LABEL in
    # the gap between the ST pair and the KL pair below it).
    ST_LABEL_Y = +0.41
    ST_SN_Y    = +0.24
    ST_FK_Y    = +0.08
    KL_LABEL_Y = -0.09
    KL_SN_Y    = -0.26
    KL_FK_Y    = -0.42

    def _fmt_uv(v):
        if v >= 1000:
            return f"{v:,.0f} \u00b5V"
        if v >= 10:
            return f"{v:.1f} \u00b5V"
        if v >= 1:
            return f"{v:.2f} \u00b5V"
        if v > 0:
            places = max(3, -int(math.floor(math.log10(v))) + 1)
            return f"{v:.{places}f} \u00b5V"
        return "0 \u00b5V"

    def _draw_sn_fock_pair(y_sn, y_fk, sn_uv, fock_uv, first=False):
        inset.barh(y_sn, sn_uv, height=bar_h,
                   color="#3CD46A", edgecolor="#3CD46A", alpha=0.85,
                   label="Sommerfeld-Norton (FE)" if first else None)
        inset.barh(y_fk, fock_uv, height=bar_h,
                   color="#FF3399", edgecolor="#FF3399", alpha=0.85,
                   label="P.526 Fock (GE)" if first else None)
        inset.text(sn_uv * 0.96, y_sn, _fmt_uv(sn_uv),
                   fontsize=9.0, color="#FFFFFF", va="center", ha="right",
                   fontweight="bold")
        if fock_uv >= 5.0:
            inset.text(fock_uv * 0.95, y_fk, _fmt_uv(fock_uv),
                       fontsize=9.0, color="#FFFFFF", va="center", ha="right",
                       fontweight="bold")
        else:
            inset.text(0.366, y_fk, _fmt_uv(fock_uv),
                       fontsize=9.0, color="#FFFFFF", va="center", ha="left",
                       fontweight="bold")
        inset.text(0.040, y_sn, "USABLE",
                   fontsize=7.5, color="#FFFFFF", va="center", ha="center",
                   fontweight="bold")
        fock_status = "USABLE" if fock_uv >= NOISE_UV else "UNUSABLE"
        if fock_status == "UNUSABLE":
            inset.barh(y_fk, NOISE_UV, height=bar_h,
                       facecolor="none", edgecolor="#FF3399",
                       linewidth=0.3, hatch="///", alpha=0.45, zorder=2)
        pink_end = fock_uv if fock_status == "USABLE" else NOISE_UV
        if sn_uv > pink_end:
            inset.barh(y_fk, sn_uv - pink_end, height=bar_h,
                       left=pink_end, facecolor="none", edgecolor="#FF3399",
                       linewidth=0.3, hatch="///", alpha=0.45, zorder=2)
        inset.text(0.040, y_fk, fock_status,
                   fontsize=7.5, color="#FFFFFF", va="center", ha="center",
                   fontweight="bold", zorder=5)

    y_centres = []
    for i, p in enumerate(PREDICTIONS):
        y0 = (n - 1 - i) * group_h
        is_highlight = p["d"] == HIGHLIGHT_KM
        _grp_top = ST_LABEL_Y + 0.09
        _grp_bot = KL_FK_Y - bar_h / 2 - 0.04
        if is_highlight:
            inset.add_patch(MplRect(
                (0, y0 + _grp_bot), 1, _grp_top - _grp_bot,
                transform=inset.get_yaxis_transform(),
                facecolor="#80E0FF", alpha=0.10, edgecolor="#80E0FF",
                linewidth=1.2, zorder=0))
        # ST pair first (top of group), then KL pair (below). Each sub-label
        # sits in the black space above its own pair.
        inset.text(0.025, y0 + ST_LABEL_Y,
                   f"St {p['d_st']:.0f} km \u2192 TF {p['d']:.0f} km",
                   fontsize=10.0, color="#FFB84D", ha="left", va="center",
                   fontweight="bold", zorder=30)
        _draw_sn_fock_pair(y0 + ST_SN_Y, y0 + ST_FK_Y,
                           p["sn_st"], p["fock_st"], first=(i == 0))
        inset.text(0.025, y0 + KL_LABEL_Y,
                   f"Kl {p['d_kl']:.0f} km \u2192 TF {p['d']:.0f} km",
                   fontsize=10.0, color="#FFE640", ha="left", va="center",
                   fontweight="bold", zorder=30)
        _draw_sn_fock_pair(y0 + KL_SN_Y, y0 + KL_FK_Y,
                           p["sn"], p["fock"])
        y_centres.append(y0)

    inset.set_yticks([])
    inset.set_ylim(KL_FK_Y - bar_h / 2 - 0.10,
                   (n - 1) * group_h + ST_LABEL_Y + 0.12)

    # noise-floor guide line (legend explains colours)
    inset.axvline(NOISE_UV, color="#FFFFFF", linewidth=1.0, alpha=0.85,
                  linestyle=(0, (5, 3)), zorder=4)

    inset.tick_params(axis="x", labelsize=7.5, colors="#D6D6E0", pad=1)
    inset.tick_params(axis="y", colors="#D6D6E0", length=0, pad=2)

    # Inset legend removed - the colour/line key now lives in the main
    # legend under the "Signal Strength" section.

    # Title lives in the empty black strip above the top bar group,
    # inside the plot area so it doesn't force matplotlib to rescale.

    plt.tight_layout()
    plt.savefig(os.path.join(_HERE, "graphs", f"ITU_Calc_knickebein_beam_map_telefunken_{HIGHLIGHT_KM}km.png"), dpi=140,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved ITU_Calc_knickebein_beam_map_telefunken_{HIGHLIGHT_KM}km.png")
    plt.close(fig)


for _d in [1000, 800, 700, 500, 400]:
    render(_d)
