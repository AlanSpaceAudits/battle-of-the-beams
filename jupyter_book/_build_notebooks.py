"""One-shot builder for the five chapter notebooks.

Run from this directory:
    python3 _build_notebooks.py
"""
from __future__ import annotations
import json
import pathlib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = pathlib.Path(__file__).parent


def write(path: pathlib.Path, cells: list) -> None:
    nb = new_notebook(cells=cells)
    nb["metadata"] = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    with path.open("w", encoding="utf-8") as fp:
        nbformat.write(nb, fp)
    print(f"wrote {path}")


# =====================================================================
# 000 — Constants & tables
# =====================================================================
def build_000():
    cells = []
    cells.append(new_markdown_cell(r"""# 000 — Constants & Tables

Everything hardcoded in `ITU_CERTIFIED_Battle_of_the_Beams_Calc2_v9_1_1.xlsx`
lives here. The remaining chapters import this module (`common.py`)
and reference these tables by name.

The mapping from spreadsheet sheets to Python objects:

| Sheet | Object |
|---|---|
| `const`        | module-level constants (`EPS_0`, `MU_0`, `C_VAC`, `R_EARTH`, `K_REFRAC`, `T_SYS`, `RX_NF_DB`, `RX_BW_HZ`) |
| `ground`       | `GROUND` dict (sigma, eps_r per ground type) |
| `stations`     | `STATIONS` dict (Kleve, Stollberg, Greny, Beaumont-Hague) |
| `targets`      | `TARGETS` dict (Midlands cities + Telefunken sea tests) |
| `paths_refs`   | reproduced inline below; haversine distance is computed from `STATIONS` + `TARGETS` |
| `Squint Sandbox` | `SANDBOX` dict (squint 5°, W = 99 m, H = 20 m) |
"""))

    cells.append(new_code_cell("""import math, pandas as pd
from IPython.display import Markdown, display
import common as c
"""))

    cells.append(new_markdown_cell(r"""## Physical constants (`const` sheet)
"""))

    cells.append(new_code_cell("""pd.DataFrame([
    ("eps_0",         c.EPS_0,         "F/m"),
    ("mu_0",          c.MU_0,          "H/m"),
    ("c (= 1/sqrt(eps_0 mu_0))", c.C_VAC, "m/s"),
    ("pi",            math.pi,         ""),
    ("k_Boltzmann",   c.K_BOLTZ,       "J/K"),
    ("R_Earth",       c.R_EARTH,       "m"),
    ("k (4/3 refraction)", c.K_REFRAC, ""),
    ("R_effective = k*R_Earth", c.R_EFFECTIVE, "m"),
    ("T_sys",         c.T_SYS,         "K"),
    ("RX noise figure NF", c.RX_NF_DB, "dB"),
    ("RX bandwidth B",     c.RX_BW_HZ, "Hz"),
    ("eta_0",         c.ETA_0,         "ohm"),
], columns=["Symbol","Value","Unit"]).style.format({"Value": "{:.6g}"})
"""))

    cells.append(new_markdown_cell(r"""## Ground electrical properties (`ground` sheet)

These feed the Fock $\beta$ via ITU-R P.526-16 Eqs. 16 / 16a and the
Sommerfeld-Norton complex permittivity $n^2 = \varepsilon_r - jx$ with
$x = 18\,000\,\sigma / f_{\text{MHz}}$.
"""))

    cells.append(new_code_cell("""pd.DataFrame([
    (k, v["sigma"], v["eps_r"], v["note"])
    for k, v in c.GROUND.items()
], columns=["ground_type","sigma (S/m)","eps_r","notes"])
"""))

    cells.append(new_markdown_cell(r"""## Stations (`stations` sheet)
"""))

    cells.append(new_code_cell("""pd.DataFrame([
    dict(name=k, **v) for k, v in c.STATIONS.items()
])
"""))

    cells.append(new_markdown_cell(r"""## Targets (`targets` sheet)
"""))

    cells.append(new_code_cell("""pd.DataFrame([
    dict(name=k, **v) for k, v in c.TARGETS.items()
])
"""))

    cells.append(new_markdown_cell(r"""## Path-distance table (`paths_refs` sheet, recomputed)

Distances are recomputed by haversine from the station lat/lon and
target lat/lon so the table stays in sync with the source data.
"""))

    cells.append(new_code_cell("""rows = []
TX = {
    'Kleve': ['Spalding','Retford','Derby','Birmingham','TF 400 km','TF 500 km','TF 700 km','TF 800 km','TF 1000 km'],
    'Stollberg': ['Beeston','Derby','Birmingham','Liverpool','TF 400 km','TF 500 km','TF 700 km','TF 800 km','TF 1000 km'],
    'Greny': ['London'],
    'Beaumont-Hague': ['London','Cardiff','Plymouth'],
}
for tx, targets in TX.items():
    s = c.STATIONS[tx]
    for tgt in targets:
        t = c.TARGETS[tgt]
        d_km = c.great_circle_m(s['lat_deg'], s['lon_deg'], t['lat'], t['lon']) / 1000
        ground = 'sea' if tgt.startswith('TF') else s['ground']
        rows.append((tx, tgt, round(d_km, 1), ground))
pd.DataFrame(rows, columns=['tx_station','target','d_km','ground'])
"""))

    cells.append(new_markdown_cell(r"""## Squint Sandbox defaults

The Squint Sandbox sheet in the workbook is where the British-measured
400 to 500 yard equisignal corridor at Spalding is calibrated. The
default values below recover the operational corridor:

| Input | Default | Symbol |
|---|---|---|
| Squint angle | 5° | $\theta$ |
| Aperture width | 99 m | $W$ |
| Aperture height | 20 m | $H$ |

Sources: Trenkle 1979 p. 67 for the 99 m sub-array array width;
Telefunken 5° squint per Bauer 2004 p. 12. The 20 m aperture-H
default is the Squint Sandbox sheet cell B18.
"""))

    cells.append(new_code_cell("""pd.DataFrame([
    ("Squint theta (deg)", c.SANDBOX['squint_deg']),
    ("Aperture W (m)",     c.SANDBOX['W_m']),
    ("Aperture H (m)",     c.SANDBOX['H_m']),
], columns=["Sandbox input","Default"])
"""))

    cells.append(new_markdown_cell(r"""## Derived constants at Knickebein parameters

These follow algebraically from the table above and the constants and
are quoted everywhere downstream.
"""))

    cells.append(new_code_cell(r"""f_MHz = 31.5
W, H = 99.0, 29.0           # ITU sheet uses 29 m for aperture H in stations
lam = c.freq_to_wavelen(f_MHz)
k_wave = c.wavenumber(f_MHz)
G_tx_dBi = c.aperture_gain_dBi(W, H, f_MHz)
P_tx_dBW = 10*math.log10(3000)
N_dBW = c.noise_floor_dBW(f_MHz)
Fa_dB = c.galactic_Fa_dB(f_MHz)
thermal_dBW = 10*math.log10(c.K_BOLTZ*c.T_SYS*c.RX_BW_HZ)
display(Markdown(rf'''
- $\lambda = c/f$ = **{lam:.4f} m**
- $k = 2\pi/\lambda$ = **{k_wave:.4f} rad/m**
- Aperture area $A = W H$ = {W*H:.0f} m²
- Aperture directivity $G_{{tx}} = 4\pi A/\lambda^2$ = **{G_tx_dBi:.2f} dBi**
- $P_{{tx}} = 10 \log_{{10}}(3000)$ = **{P_tx_dBW:.3f} dBW**
- Thermal $kTB$ at 290 K, 500 Hz = **{thermal_dBW:.3f} dBW**
- Galactic $F_a = 52 - 23 \log_{{10}}(f_{{MHz}})$ at 31.5 MHz = **{Fa_dB:.2f} dB**
- Noise floor $N = kTB \cdot \max(NF, F_a)$ = **{N_dBW:.3f} dBW**, equivalent to
  **{c.voltage_50ohm_uV(N_dBW)*1000:.2f} nV** at the 50 ohm input.
'''))
"""))

    cells.append(new_markdown_cell(r"""---
The next four chapters use this module unchanged. They differ only in
which transmitter and which propagation model is applied.
"""))

    write(HERE / "000_Constants.ipynb", cells)


# =====================================================================
# Fock GE chapters (001 = Kleve, 002 = Stollberg)
# =====================================================================
def build_fock(station: str, chapter: str, default_target: str, default_ground_note: str):
    cells = []
    pretty = "director beam (Kleve)" if station == "Kleve" else "cross beam (Stollberg)"
    row_no = "row 2" if station == "Kleve" else "row 3"
    cells.append(new_markdown_cell(
        f"# {chapter} — Fock smooth-Earth diffraction (globe), {station}\n\n"
        f"ITU-R P.526-16 §3 Fock first-term residue series. Mirrors ITU sheet\n"
        f"columns V through BP for {row_no}.\n\n"
        "This is the **globe (GE) model** — exponential field-strength decay\n"
        "past the geometric horizon caused by the wave bending around the\n"
        "sphere of radius 6 371 km. Every Knickebein path past the radio\n"
        "horizon enters the residue-series shadow zone, and the loss in that\n"
        "zone grows linearly with distance.\n\n"
        f"The {pretty} runs from {station} over {default_ground_note} to a\n"
        "6 000 m He 111 receiver. Change `target` in the next cell to any\n"
        "name from `TARGETS` and re-run.\n"
    ))

    cells.append(new_code_cell(
        "import math, cmath, pandas as pd\n"
        "from IPython.display import Markdown, display\n"
        "import common as c\n\n"
        f"station = {station!r}\n"
        f"target  = {default_target!r}      # change me\n\n"
        "s = c.STATIONS[station]\n"
        "t = c.TARGETS[target]\n"
        "f_MHz = s['freq_MHz']\n"
        "ground = 'sea' if target.startswith('TF') else s['ground']\n"
    ))

    cells.append(new_markdown_cell(r"""## Geometry on a globe

Great-circle distance via haversine (ITU sheet cell `O2` / `O3`):
$$d = 2 R_\oplus \arcsin\sqrt{\sin^2\!\tfrac{\Delta\varphi}{2} + \cos\varphi_1 \cos\varphi_2 \sin^2\!\tfrac{\Delta\lambda}{2}}$$

Effective Earth radius (4/3 model, ITU-R P.453):
$$a_e = \tfrac{4}{3} R_\oplus = 8\,494\,667\ \text{m}$$

Radio horizon, each end:
$$d_{\mathrm{LoS},\mathrm{tx}} = \sqrt{2 a_e h_{tx}}, \qquad d_{\mathrm{LoS},\mathrm{rx}} = \sqrt{2 a_e h_{rx}}$$

If $d < d_{\mathrm{LoS},\mathrm{tx}} + d_{\mathrm{LoS},\mathrm{rx}}$ the path is line-of-sight and Fock returns zero diffraction loss. Otherwise the path is in the diffraction shadow zone.
"""))

    cells.append(new_code_cell("""d_m  = c.great_circle_m(s['lat_deg'], s['lon_deg'], t['lat'], t['lon'])
h_tx = s['h_tx_m']
h_rx = t['rx_alt_m']
d_los_tx = math.sqrt(2*c.R_EFFECTIVE*h_tx)
d_los_rx = math.sqrt(2*c.R_EFFECTIVE*h_rx)
d_los    = d_los_tx + d_los_rx
in_shadow = d_m > d_los
display(Markdown(f'''
- $d$ = **{d_m/1000:.2f} km**
- $h_{{tx}}$ = {h_tx} m, $h_{{rx}}$ = {h_rx} m
- $d_{{LoS,tx}}$ = {d_los_tx/1000:.2f} km
- $d_{{LoS,rx}}$ = {d_los_rx/1000:.2f} km
- $d_{{LoS}}$ total = **{d_los/1000:.2f} km**
- shadow zone? **{in_shadow}**, shadow length = {(d_m-d_los)/1000 if in_shadow else 0:.2f} km
'''))
"""))

    cells.append(new_markdown_cell(r"""## $\beta$ polarisation parameter (ITU-R P.526-16 Eqs. 16, 16a)

The Fock $\beta$ is unity for horizontal polarisation at any frequency,
unity for vertical polarisation over land above 20 MHz, and unity for
vertical polarisation over sea above 300 MHz. Otherwise it is computed
from the surface admittance $K$:

$$K^2 \approx \frac{6.89\,\sigma}{k^{2/3}\,f_{MHz}^{5/3}} \qquad \text{(Eq. 16a)}$$

$$\beta = \frac{1 + 1.6 K^2 + 0.67 K^4}{1 + 4.5 K^2 + 1.53 K^4} \qquad \text{(Eq. 16)}$$

Knickebein at 31.5 MHz vertical pol falls into the rule where:

- over **land** (above the 20 MHz cut), $\beta \to 1$;
- over **sea**  (below the 300 MHz cut), $\beta$ is taken from Eq. 16
  using $K$ from Eq. 16a. For sea $\sigma = 5$ S/m the result is
  $\beta \approx 0.81$, and $K$ propagates into the lower-bound clamp on
  $G(Y)$ via Eq. 18.
"""))

    cells.append(new_code_cell(r"""beta, K_floor = c.p526_beta_and_K(ground, s['pol'], f_MHz, c.K_REFRAC)
sigma = c.GROUND[ground]['sigma']
K2 = 6.89*sigma / (c.K_REFRAC**(2/3) * f_MHz**(5/3))
K  = math.sqrt(K2)
beta_eq16 = (1 + 1.6*K2 + 0.67*K2**2) / (1 + 4.5*K2 + 1.53*K2**2)
display(Markdown(rf'''
- ground = **{ground}**, sigma = {sigma} S/m
- $K^2$ = {K2:.4g}, $K$ = {K:.4f}
- $\beta$ from Eq. 16 = {beta_eq16:.4f}
- $\beta$ final = **{beta:.4f}** (over-ride to 1 above frequency cut)
- $K$ used in $G(Y)$ floor = {K_floor:.4f}
'''))
"""))

    cells.append(new_markdown_cell(r"""## Normalised distance and heights

$$X = \beta \left(\frac{\pi}{\lambda a_e^2}\right)^{1/3} d \qquad \text{(Eq. 13)}$$
$$Y_{1,2} = 2 \beta \left(\frac{\pi^2}{\lambda^2 a_e}\right)^{1/3} h_{tx,rx} \qquad \text{(Eq. 13)}$$
"""))

    cells.append(new_code_cell("""X  = c.p526_X(d_m, beta, f_MHz)
Y1 = c.p526_Y(h_tx, beta, f_MHz)
Y2 = c.p526_Y(h_rx, beta, f_MHz)
display(Markdown(f'''
- $X$ = **{X:.3f}**  (long-path branch if X >= 1.6)
- $Y_1$ (TX) = **{Y1:.3f}**
- $Y_2$ (RX) = **{Y2:.3f}**
- long-path branch? **{X >= 1.6}**
'''))
"""))

    cells.append(new_markdown_cell(r"""## Distance term $F(X)$ (Eq. 14 / 15)

$$F(X) = \begin{cases}
11 + 10\log_{10} X - 17.6\,X & X \ge 1.6 \\
-20\log_{10} X - 5.6488\,X^{1.425} & X < 1.6
\end{cases}$$

## Height-gain term $G(Y)$ (Eq. 17, with Eq. 18 floor)

$$G(Y) = \begin{cases}
17.6\sqrt{B - 1.1} - 5\log_{10}(B - 1.1) - 8 & B > 2 \\
20\log_{10}\bigl(B + 0.1 B^3\bigr) & B \le 2
\end{cases}, \quad B = \beta Y$$

$$G(Y) \ge 2 + 20 \log_{10} K \quad \text{(Eq. 18 lower bound, when sea)}$$

Field strength relative to free space:
$$E/E_0\ \text{(dB)} = F(X) + G(Y_1) + G(Y_2)$$

Diffraction loss is the negative of this, clipped at zero inside the
line of sight.
"""))

    cells.append(new_code_cell("""F_X  = c.F_of_X(X)
B1, B2 = beta*Y1, beta*Y2
G_Y1 = c.G_of_Y(B1, K_floor)
G_Y2 = c.G_of_Y(B2, K_floor)
E_over_E0 = F_X + G_Y1 + G_Y2
L_diff = c.fock_diff_loss_dB(d_m, h_tx, h_rx, f_MHz, ground, s['pol'])
display(Markdown(f'''
- $F(X)$  = **{F_X:.3f} dB**
- $G(Y_1)$ = {G_Y1:.3f} dB
- $G(Y_2)$ = {G_Y2:.3f} dB
- $E/E_0$  = **{E_over_E0:.3f} dB**
- diffraction loss applied = **{L_diff:.3f} dB**
'''))
"""))

    cells.append(new_markdown_cell(r"""## Link budget (ITU sheet cols AS-BP)

$$\text{FSPL} = 20 \log_{10}\!\frac{4\pi d}{\lambda}, \qquad G_{tx} = 10\log_{10}\!\frac{4\pi A}{\lambda^2}$$

$$P_{rx} = P_{tx} + G_{tx} + G_{rx} - \text{FSPL} - L_{\text{diffraction}}$$

Galactic noise (ITU-R P.372-16 Eq. 14, $f < 100$ MHz):
$$F_a = 52 - 23 \log_{10} f_{MHz}$$

Noise floor at the 50 ohm input:
$$N = 10\log_{10}(k T_{sys} B) + \max(NF, F_a)$$

Crossover loss (5° squint of the 99 m aperture):
$$L_{\text{cross}} = 20\log_{10}\!\left|\frac{\sin(\pi W \sin\theta/\lambda)}{\pi W \sin\theta/\lambda}\right|$$

The pilot flies the equisignal corridor, not the sub-beam peak, so the
equisignal SNR is $\text{SNR}_{\text{peak}} + L_{\text{cross}}$ (the
loss is negative).
"""))

    cells.append(new_code_cell("""r = c.link_budget(station, target, model='fock')
pd.DataFrame({
    'Quantity':[
        'FSPL (dB)',
        'G_tx (dBi)',
        'P_tx (dBW)',
        'P_rx (dBW)',
        'Noise floor (dBW)',
        'SNR peak (dB)',
        'Crossover at 5 deg (dB)',
        'SNR equisignal (dB)',
        'V_eq at 50 ohm (uV)',
        'V_noise at 50 ohm (uV)',
    ],
    'Value':[
        r['FSPL_dB'], r['G_tx_dBi'], r['P_tx_dBW'],
        r['P_rx_dBW'], r['N_dBW'], r['SNR_peak_dB'],
        r['crossover_dB'], r['SNR_eq_dB'],
        r['V_eq_uV'], r['V_noise_uV'],
    ],
}).style.format({'Value': '{:.4g}'})
"""))

    cells.append(new_markdown_cell(r"""## Verdict against the +10 dB detection floor

The 0.079 uV physics noise floor at the 50 ohm input is the bare
detection threshold once the FuBl 2 / EBL 3 receiver AGC is taken
into account. The +10 dB margin is the standard bare-RF detection
floor above noise.
"""))

    cells.append(new_code_cell("""verdict = ('PASS' if r['SNR_eq_dB'] >= 10
           else 'MARGINAL' if r['SNR_eq_dB'] >= 0
           else 'FAIL')
display(Markdown(f'''
- SNR equisignal = **{r['SNR_eq_dB']:+.2f} dB**
- V equisignal   = **{r['V_eq_uV']:.4g} uV**  (noise floor {r['V_noise_uV']:.4g} uV)
- verdict = **{verdict}**
'''))
"""))

    cells.append(new_markdown_cell(rf"""## Sweep all confirmed {station} paths

This is the table the spreadsheet's Main / ITU rows produce when the
target dropdown is cycled.
"""))

    target_list = (
        ['Spalding','Retford','Derby','Birmingham','TF 400 km','TF 500 km','TF 700 km','TF 800 km','TF 1000 km']
        if station == "Kleve" else
        ['Beeston','Derby','Birmingham','Liverpool','TF 400 km','TF 500 km','TF 700 km','TF 800 km','TF 1000 km']
    )
    cells.append(new_code_cell(f"""rows = []
for tgt in {target_list!r}:
    rr = c.link_budget(station, tgt, model='fock')
    v = ('PASS' if rr['SNR_eq_dB'] >= 10
         else 'MARGINAL' if rr['SNR_eq_dB'] >= 0
         else 'FAIL')
    rows.append((tgt, round(rr['d_km'],1), rr['ground'],
                 round(rr['diffraction_loss_dB'],2),
                 round(rr['SNR_peak_dB'],2),
                 round(rr['SNR_eq_dB'],2),
                 round(rr['V_eq_uV'],4), v))
pd.DataFrame(rows, columns=['target','d_km','ground','L_diff_dB',
                            'SNRpeak_dB','SNReq_dB','V_eq_uV','verdict'])
"""))

    cells.append(new_markdown_cell(r"""## Squint Sandbox — edit and observe

The Squint Sandbox sheet in the workbook is the calibration cell.
Edit `squint_deg`, `W_m`, `H_m` below and re-run to see how the
equisignal corridor width and the equisignal SNR move.

The defaults (`squint = 5 deg, W = 99 m, H = 20 m`) reproduce the
400 to 500 yard equisignal corridor at Spalding measured by R/T flight
F/Lt Bufton on 21 June 1940.
"""))

    cells.append(new_code_cell("""squint_deg = c.SANDBOX['squint_deg']   # default 5
W_m        = c.SANDBOX['W_m']          # default 99
H_m        = c.SANDBOX['H_m']          # default 20

lam = c.freq_to_wavelen(f_MHz)
u  = math.pi*W_m*math.sin(math.radians(squint_deg))/lam
L_cross = 20*math.log10(abs(math.sin(u)/u))
G_sub_dBi  = 10*math.log10(4*math.pi*W_m*H_m / lam**2)
width_m = c.equisignal_corridor_width_m(d_m, W_m, squint_deg, f_MHz)
SNR_eq  = r['SNR_peak_dB'] + L_cross
display(Markdown(f'''
- u = pi W sin(theta)/lambda = {u:.4f}
- crossover loss = **{L_cross:.3f} dB**
- aperture directivity (W={W_m} m, H={H_m} m) = **{G_sub_dBi:.2f} dBi**
- equisignal corridor width at d = {d_m/1000:.1f} km
  -> **{width_m:.1f} m** = **{width_m/0.9144:.1f} yards**
- SNR equisignal (using peak from above) = **{SNR_eq:+.2f} dB**
'''))
"""))

    write(HERE / f"{chapter}_Fock_GE_{('KL' if station == 'Kleve' else 'ST')}.ipynb", cells)


# =====================================================================
# Sommerfeld FE chapters (003 = Kleve, 004 = Stollberg)
# =====================================================================
def build_sommerfeld(station: str, chapter: str, default_target: str, default_ground_note: str):
    cells = []
    pretty = "director beam (Kleve)" if station == "Kleve" else "cross beam (Stollberg)"
    cells.append(new_markdown_cell(rf"""# {chapter} — Sommerfeld-Norton FE three-term Ez, {station}

ITU Handbook on Ground Wave Propagation (2014) Part 1 §3.2.1. Mirrors
ITU sheet columns BQ through CY for {"row 6" if station == "Kleve" else "row 7"}.

This is the **flat-earth (FE) model**. There is no curvature, no
horizon, no shadow zone. The field at the receiver is the coherent sum
of three contributions:

1. direct ray $E_d$ along the geometric line $r_1$,
2. ground-reflected ray $E_r$ along $r_2$ with Fresnel reflection coefficient $R_v$,
3. Norton surface wave $E_s$ with attenuation function $F$.

$$E_z = E_d + E_r + E_s$$

The {pretty} runs from {station} over {default_ground_note} to a
6 000 m He 111 receiver. Change `target` in the next cell to any
name from `TARGETS` and re-run.
"""))

    cells.append(new_code_cell(f"""import math, cmath, pandas as pd
from IPython.display import Markdown, display
import common as c

station = {station!r}
target  = {default_target!r}      # change me

s = c.STATIONS[station]
t = c.TARGETS[target]
f_MHz = s['freq_MHz']
ground = 'sea' if target.startswith('TF') else s['ground']
sigma = c.GROUND[ground]['sigma']
eps_r = c.GROUND[ground]['eps_r']
"""))

    cells.append(new_markdown_cell(r"""## Geometry on a flat earth

Straight-line slant ranges to the receiver (no curvature):

$$r_1 = \sqrt{d^2 + (h_{rx} - h_{tx})^2}, \qquad r_2 = \sqrt{d^2 + (h_{rx} + h_{tx})^2}$$

Grazing angles $\psi_1$, $\psi_2$ measured from the surface:

$$\cos^2\psi_1 = (d/r_1)^2, \quad \cos^2\psi_2 = (d/r_2)^2, \quad \sin\psi_2 = (h_{rx} + h_{tx})/r_2$$

The receiver image is the ground reflection point of the source.
"""))

    cells.append(new_code_cell("""d_m  = c.great_circle_m(s['lat_deg'], s['lon_deg'], t['lat'], t['lon'])
h_tx = s['h_tx_m']
h_rx = t['rx_alt_m']
lam  = c.freq_to_wavelen(f_MHz)
k    = c.wavenumber(f_MHz)
r1 = math.sqrt(d_m**2 + (h_rx - h_tx)**2)
r2 = math.sqrt(d_m**2 + (h_rx + h_tx)**2)
cos2_psi1 = (d_m/r1)**2
cos2_psi2 = (d_m/r2)**2
sin_psi2  = (h_rx+h_tx)/r2
display(Markdown(f'''
- d = {d_m/1000:.2f} km, lambda = {lam:.4f} m, k = {k:.4f} rad/m
- r1 = {r1:.2f} m, r2 = {r2:.2f} m
- cos^2 psi1 = {cos2_psi1:.6f}, cos^2 psi2 = {cos2_psi2:.6f}
- sin psi2 = {sin_psi2:.6f}
'''))
"""))

    cells.append(new_markdown_cell(r"""## Complex ground impedance

$$x = \frac{18\,000\,\sigma}{f_{MHz}}, \qquad n^2 = \varepsilon_r - j\,x, \qquad u^2 = \frac{2}{n^2}$$

Fresnel reflection coefficient (vertical polarisation):

$$R_v = \frac{n^2 \sin\psi_2 - \sqrt{n^2 - \cos^2\psi_2}}{n^2 \sin\psi_2 + \sqrt{n^2 - \cos^2\psi_2}}$$
"""))

    cells.append(new_code_cell("""x  = 18000.0*sigma/f_MHz
n2 = complex(eps_r, -x)
u2 = 2.0/n2
sqrt_term = cmath.sqrt(n2 - cos2_psi2)
Rv = (n2*sin_psi2 - sqrt_term) / (n2*sin_psi2 + sqrt_term)
display(Markdown(f'''
- x = 18000 sigma / f = {x:.4f}
- n^2 = {n2}
- u^2 = {u2}
- sqrt(n^2 - cos^2 psi2) = {sqrt_term}
- R_v = {Rv}
'''))
"""))

    cells.append(new_markdown_cell(r"""## Numerical distance $w$ and attenuation function $F$

Norton's numerical distance for vertical polarisation:

$$w = \frac{-j\,2\,k\,r_2\,u^2\,(1 - u^2 \cos^2\psi_2)}{1 - R_v}$$

For large $|w|$ the attenuation function is well approximated by its
asymptotic expansion (ITU Handbook Part 1 §3.2.1 Eq. 7):

$$F \approx -\frac{1}{2w} - \frac{3}{(2w)^2} - \frac{15}{(2w)^3} - \frac{105}{(2w)^4}$$

This is exactly the formula used by ITU sheet column CJ.
"""))

    cells.append(new_code_cell("""one_minus_Rv = 1 - Rv
one_minus_u2_cos2 = 1 - u2*cos2_psi2
minus_j_2kr2 = complex(0.0, -2*k*r2)
w_num = minus_j_2kr2 * u2 * one_minus_u2_cos2
w = w_num / one_minus_Rv
two_w = 2*w
F = (-1/two_w) + (-3/two_w**2) + (-15/two_w**3) + (-105/two_w**4)
display(Markdown(f'''
- w = {w}
- |w| = {abs(w):.4g}
- F = {F}
- |F| = {abs(F):.4g}
'''))
"""))

    cells.append(new_markdown_cell(r"""## Three-term superposition

Each term has its own propagator $\exp(-j k r)$ and aperture factor:

$$E_d = \frac{\cos^2\psi_1}{r_1} e^{-j k r_1}$$
$$E_r = \frac{\cos^2\psi_2}{r_2}\, R_v\, e^{-j k r_2}$$
$$E_s = \frac{(1 - R_v)(1 - u^2 + u^4 \cos^2\psi_2)\,F}{r_2} e^{-j k r_2}$$

Total complex field (normalised per unit source strength):

$$E_z = E_d + E_r + E_s$$
"""))

    cells.append(new_code_cell("""exp_jkr1 = cmath.exp(complex(0, -k*r1))
exp_jkr2 = cmath.exp(complex(0, -k*r2))
direct  = cos2_psi1 * exp_jkr1 / r1
reflect = cos2_psi2 * Rv * exp_jkr2 / r2
u4 = u2**2
bracket = 1 - u2 + u4*cos2_psi2
surface = one_minus_Rv * bracket * F * exp_jkr2 / r2
Ez_sum = direct + reflect + surface
display(Markdown(f'''
- E_direct  = {direct}
- E_reflect = {reflect}
- E_surface = {surface}
- E_z sum   = {Ez_sum}
- |E_z|     = {abs(Ez_sum):.6g}
'''))
"""))

    cells.append(new_markdown_cell(r"""## Absolute field strength and link budget

The ITU sheet normalises to $\sqrt{90 P_{tx}}$ for the short-dipole
reference, then boosts by the 99x29 m aperture directivity (less the
1.5 baseline of a half-wave dipole):

$$|E_z|_{abs} = \sqrt{90\,P_{tx}} \cdot |E_z|_{sum}$$
$$E_{boost} = |E_z|_{abs} \cdot \sqrt{G_{tx,lin} / 1.5}$$

Received power at an isotropic receive antenna:

$$P_{rx} = \frac{E_{boost}^2 \lambda^2}{8\pi \eta_0}, \qquad \eta_0 = 376.73\ \Omega$$
"""))

    cells.append(new_code_cell("""sqrt_90P = math.sqrt(90 * s['Ptx_W'])
G_tx_dBi = c.aperture_gain_dBi(s['W_m'], s['H_m'], f_MHz)
G_tx_lin = 10**(G_tx_dBi/10)
Ez_Vpm   = sqrt_90P * abs(Ez_sum)
E_boost  = Ez_Vpm * math.sqrt(G_tx_lin / 1.5)
P_rx_W   = E_boost**2 * lam**2 / (8*math.pi*c.ETA_0)
P_rx_dBW = 10*math.log10(P_rx_W)
display(Markdown(f'''
- sqrt(90 P_tx) = {sqrt_90P:.3f}
- G_tx = {G_tx_dBi:.2f} dBi   (linear {G_tx_lin:.2f})
- |E_z| absolute = {Ez_Vpm:.6g} V/m
- E_boost        = {E_boost:.6g} V/m
- P_rx           = {P_rx_W:.4g} W
- P_rx           = **{P_rx_dBW:.3f} dBW**
'''))
"""))

    cells.append(new_markdown_cell(r"""## Equisignal SNR and verdict

Crossover and noise floor follow the same formulas as the Fock
chapter. The flat-earth model has no diffraction loss, so the
equisignal SNR scales only with FSPL plus the squint crossover.
"""))

    cells.append(new_code_cell("""r = c.link_budget(station, target, model='sommerfeld')
verdict = ('PASS' if r['SNR_eq_dB'] >= 10
           else 'MARGINAL' if r['SNR_eq_dB'] >= 0
           else 'FAIL')
pd.DataFrame({
    'Quantity':[
        'P_rx (dBW)',
        'Noise floor (dBW)',
        'SNR peak (dB)',
        'Crossover at 5 deg (dB)',
        'SNR equisignal (dB)',
        'V_eq at 50 ohm (uV)',
        'V_noise at 50 ohm (uV)',
        'Verdict',
    ],
    'Value':[
        r['P_rx_dBW'], r['N_dBW'], r['SNR_peak_dB'],
        r['crossover_dB'], r['SNR_eq_dB'],
        r['V_eq_uV'], r['V_noise_uV'], verdict,
    ],
})
"""))

    cells.append(new_markdown_cell(rf"""## Sweep all confirmed {station} paths
"""))

    target_list = (
        ['Spalding','Retford','Derby','Birmingham','TF 400 km','TF 500 km','TF 700 km','TF 800 km','TF 1000 km']
        if station == "Kleve" else
        ['Beeston','Derby','Birmingham','Liverpool','TF 400 km','TF 500 km','TF 700 km','TF 800 km','TF 1000 km']
    )
    cells.append(new_code_cell(f"""rows = []
for tgt in {target_list!r}:
    rr = c.link_budget(station, tgt, model='sommerfeld')
    v = ('PASS' if rr['SNR_eq_dB'] >= 10
         else 'MARGINAL' if rr['SNR_eq_dB'] >= 0
         else 'FAIL')
    rows.append((tgt, round(rr['d_km'],1), rr['ground'],
                 round(rr['P_rx_dBW'],2),
                 round(rr['SNR_peak_dB'],2),
                 round(rr['SNR_eq_dB'],2),
                 round(rr['V_eq_uV'],3), v))
pd.DataFrame(rows, columns=['target','d_km','ground','P_rx_dBW',
                            'SNRpeak_dB','SNReq_dB','V_eq_uV','verdict'])
"""))

    write(HERE / f"{chapter}_Sommerfeld_FE_{('KL' if station == 'Kleve' else 'ST')}.ipynb", cells)


if __name__ == "__main__":
    build_000()
    build_fock("Kleve",     "001", "Derby",   "land (sigma = 0.005 S/m, eps_r = 15)")
    build_fock("Stollberg", "002", "Beeston", "sea (sigma = 5 S/m, eps_r = 70, beta = 0.81)")
    build_sommerfeld("Kleve",     "003", "Derby",   "land (sigma = 0.005 S/m, eps_r = 15)")
    build_sommerfeld("Stollberg", "004", "Beeston", "sea (sigma = 5 S/m, eps_r = 70)")
