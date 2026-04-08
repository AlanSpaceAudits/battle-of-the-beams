#!/usr/bin/env python3
"""
Battle of the Beams — SNR Bar Graphs
=====================================

Generates bar charts comparing Flat vs Globe SNR at beam peak and
equisignal crossover for both paths (Kleve and Stollberg).

The noise floor is shown as a horizontal line at 0 dB SNR.
Bars above the line = detectable. Bars below = undetectable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.special import airy

# ================================================================
#  Constants (from botb_propagation.py)
# ================================================================
C       = 299_792_458.0
R_EARTH = 6_371_000.0
K_REFR  = 4.0 / 3.0
R_EFF   = K_REFR * R_EARTH
K_BOLTZ = 1.380649e-23

FREQ     = 31_500_000.0
LAM      = C / FREQ
K_WAVE   = 2.0 * np.pi / LAM
P_TX     = 3000.0
L_H      = 99.0
H_V      = 29.0
A_PHYS   = L_H * H_V
G_DIR    = 4.0 * np.pi * A_PHYS / LAM**2
G_DIR_dB = 10.0 * np.log10(G_DIR)

H_TX       = 200.0
H_AIRCRAFT = 6000.0
RX_NF_dB   = 10.0
RX_BW_Hz   = 3000.0
T_SYS      = 290.0
FA_dB      = 32.0

CROSSOVER_DB = -19.0  # equisignal crossover for ~500 yd beam

# ================================================================
#  Functions
# ================================================================
def fock_loss(d, h_tx, h_rx, R=R_EFF):
    kR  = K_WAVE * R
    m   = (kR / 2.0) ** (1.0 / 3.0)
    xi  = m * d / R
    Y1  = 2.0 * m**2 * h_tx / R
    Y2  = 2.0 * m**2 * h_rx / R
    tau = np.array([2.33811, 4.08795, 5.52056, 6.78671, 7.94417])
    ej  = np.exp(1j * np.pi / 3.0)
    V_sum = 0.0 + 0.0j
    for ts in tau:
        _, aip, _, _ = airy(-ts)
        wt  = 1.0 / aip**2
        arg = 1j * ej * ts * xi
        V_sum += np.exp(arg) * wt
    phase_factor = np.exp(1j * np.pi / 4.0)
    V  = phase_factor * 2.0 * np.sqrt(np.pi * max(xi, 1e-30)) * V_sum
    V_abs = abs(V)
    def hgain(Y):
        return np.sqrt(1.0 + np.pi * Y)
    G1 = hgain(Y1)
    G2 = hgain(Y2)
    F  = V_abs * G1 * G2
    loss_dB = -20.0 * np.log10(max(abs(F), 1e-300))
    return loss_dB

def link_budget(d, diff_loss_dB, rx_gain_dBi=3.0):
    fspl = 20.0 * np.log10(4.0 * np.pi * d / LAM)
    P_rx_dBW = (10.0 * np.log10(P_TX) + G_DIR_dB + rx_gain_dBi
                - fspl - diff_loss_dB)
    N_thermal = 10.0 * np.log10(K_BOLTZ * T_SYS * RX_BW_Hz)
    N_total   = N_thermal + max(RX_NF_dB, FA_dB)
    SNR = P_rx_dBW - N_total
    return SNR


# ================================================================
#  Compute values for both paths
# ================================================================
paths = [
    {"name": "Kleve → Spalding",    "short": "Kleve",    "d": 439_541.0},
    {"name": "Stollberg → Beeston", "short": "Stollberg", "d": 693_547.0},
]

for p in paths:
    d = p["d"]
    fock_dB = fock_loss(d, H_TX, H_AIRCRAFT)
    p["flat_peak"]    = link_budget(d, 0.0)
    p["flat_eqsig"]   = p["flat_peak"] + CROSSOVER_DB
    p["globe_peak"]   = link_budget(d, fock_dB)
    p["globe_eqsig"]  = p["globe_peak"] + CROSSOVER_DB
    p["fock_loss"]     = fock_dB
    p["d_km"]          = d / 1000.0


# ================================================================
#  Style
# ================================================================
plt.style.use('dark_background')
FLAT_COLOR  = '#4CAF50'   # green
GLOBE_COLOR = '#F44336'   # red
NOISE_COLOR = '#FFD54F'   # yellow
BAR_ALPHA   = 0.9


# ================================================================
#  Figure 1: Side-by-side bars for both paths
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
fig.suptitle('Battle of the Beams — Flat vs Globe SNR',
             fontsize=16, fontweight='bold', color='white', y=0.97)

for ax, p in zip(axes, paths):
    labels = ['Flat\nPeak', 'Flat\nEquisignal', 'Globe\nPeak', 'Globe\nEquisignal']
    values = [p["flat_peak"], p["flat_eqsig"], p["globe_peak"], p["globe_eqsig"]]
    colors = [FLAT_COLOR, FLAT_COLOR, GLOBE_COLOR, GLOBE_COLOR]
    alphas = [0.9, 0.6, 0.9, 0.6]

    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, values, color=colors, alpha=BAR_ALPHA, width=0.6,
                  edgecolor='white', linewidth=0.5)

    # Set alpha per bar
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    # Noise floor line at 0 dB SNR
    ax.axhline(y=0, color=NOISE_COLOR, linewidth=2, linestyle='--',
               label='Noise floor (0 dB SNR)', zorder=5)

    # Detection threshold line at 10 dB
    ax.axhline(y=10, color='#FF9800', linewidth=1.5, linestyle=':',
               label='Detection threshold (10 dB)', zorder=5)

    # Value labels on bars
    for i, (val, bar) in enumerate(zip(values, bars)):
        y_offset = 2 if val >= 0 else -4
        va = 'bottom' if val >= 0 else 'top'
        ax.text(i, val + y_offset, f'{val:.1f} dB',
                ha='center', va=va, fontsize=10, fontweight='bold',
                color='white')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(f'{p["name"]}\n({p["d_km"]:.0f} km, Fock loss = {p["fock_loss"]:.0f} dB)',
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('SNR (dB)' if ax == axes[0] else '', fontsize=12)
    ax.grid(axis='y', alpha=0.2, color='gray')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.3)

# Set y-axis range to show the full picture
axes[0].set_ylim(-100, 100)

fig.tight_layout(rect=[0, 0.03, 1, 0.93])

# Add subtitle
fig.text(0.5, 0.01,
         'Bars above the yellow dashed line = signal detectable  |  '
         'Below = signal lost in galactic noise\n'
         'Equisignal = at the crossover point where the pilot flies '
         '(−19 dB from peak for ~500 yd corridor)',
         ha='center', fontsize=9, color='#AAAAAA', style='italic')

plt.savefig('/home/alan/claude/BotB/botb_snr_comparison.png', dpi=200,
            bbox_inches='tight', facecolor='#1a1a1a')
print("Saved: botb_snr_comparison.png")


# ================================================================
#  Figure 2: SNR vs Distance (continuous curve)
# ================================================================
fig2, ax2 = plt.subplots(figsize=(12, 7))

distances_km = np.arange(50, 901, 5)

flat_peak_snr  = []
globe_peak_snr = []
flat_eq_snr    = []
globe_eq_snr   = []

for d_km in distances_km:
    d = d_km * 1000.0
    fk = fock_loss(d, H_TX, H_AIRCRAFT)
    fp = link_budget(d, 0.0)
    gp = link_budget(d, fk)
    flat_peak_snr.append(fp)
    globe_peak_snr.append(gp)
    flat_eq_snr.append(fp + CROSSOVER_DB)
    globe_eq_snr.append(gp + CROSSOVER_DB)

# Plot curves
ax2.plot(distances_km, flat_peak_snr, color=FLAT_COLOR, linewidth=2.5,
         label='Flat model (beam peak)', linestyle='-')
ax2.plot(distances_km, flat_eq_snr, color=FLAT_COLOR, linewidth=2,
         label='Flat model (equisignal)', linestyle='--', alpha=0.7)
ax2.plot(distances_km, globe_peak_snr, color=GLOBE_COLOR, linewidth=2.5,
         label='Globe model (beam peak)', linestyle='-')
ax2.plot(distances_km, globe_eq_snr, color=GLOBE_COLOR, linewidth=2,
         label='Globe model (equisignal)', linestyle='--', alpha=0.7)

# Noise floor
ax2.axhline(y=0, color=NOISE_COLOR, linewidth=2, linestyle='-',
            label='Noise floor', zorder=5)
ax2.axhline(y=10, color='#FF9800', linewidth=1.5, linestyle=':',
            label='Detection threshold (10 dB)', zorder=5)

# Mark the two paths
for p in paths:
    ax2.axvline(x=p["d_km"], color='#888888', linewidth=1, linestyle=':',
                alpha=0.6)
    ax2.text(p["d_km"] + 5, 95, p["short"], fontsize=10, color='#AAAAAA',
             rotation=90, va='top')

# Shade the "undetectable" region
ax2.axhspan(-200, 0, alpha=0.08, color=NOISE_COLOR, zorder=0)
ax2.text(870, -5, 'BELOW NOISE', fontsize=8, color=NOISE_COLOR,
         ha='right', va='top', alpha=0.6)

ax2.set_xlabel('Distance (km)', fontsize=12)
ax2.set_ylabel('SNR (dB)', fontsize=12)
ax2.set_title('Knickebein SNR vs Distance — Flat vs Globe\n'
              '31.5 MHz, 3 kW, 26 dBi TX, aircraft at 6,000 m',
              fontsize=14, fontweight='bold')
ax2.set_xlim(50, 900)
ax2.set_ylim(-150, 105)
ax2.grid(alpha=0.2, color='gray')
ax2.legend(loc='lower left', fontsize=9, framealpha=0.3)

fig2.tight_layout()
plt.savefig('/home/alan/claude/BotB/botb_snr_vs_distance.png', dpi=200,
            bbox_inches='tight', facecolor='#1a1a1a')
print("Saved: botb_snr_vs_distance.png")

plt.show()
