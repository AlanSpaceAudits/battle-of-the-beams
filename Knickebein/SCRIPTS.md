# BotB Scripts — maintenance notes

Short reference for the Python pipeline in this folder. Read this before touching any of the `.py` files so you do not have to reload the full April 2026 conversation.

## Files in this folder

| File | Role |
|---|---|
| `../botb_itu_analysis.py` | Canonical physics library (repo root). Holds all constants, link-budget math, ITU P.526-16 Fock diffraction, Sommerfeld-Norton, noise-floor derivation. Everything else imports from here. |
| `../grwave/` | Vendored `grwave` package (repo root). Required by `make_p526_vs_p368_graphs.py` for P.368 ground-wave comparison curves. |
| `make_p526_vs_p368_graphs.py` | Produces the μV bar charts and Kleve/Stollberg/Telefunken sweep curves used in the null doc. Reads constants from the library. |
| `knickebein_beam_map.py` | Produces the Kleve + Stollberg beam-coverage map over the UK (operational variant, beams cross over Derby). |
| `knickebein_beam_map_telefunken.py` | Same map style, Telefunken Sep-1939 range targets along the Kleve LoS (400 / 500 / 700 / 800 / 1000 km). Emits five PNGs + four looping GIFs cycling through each highlighted target. |
| `compute_signal_strengths.py` | Writes `botb_signal_strengths.csv` with every canonical path's Friis / Sommerfeld-Norton / Fock values at both peak and equisignal. Consumers read the CSV instead of recomputing. |
| `botb_signal_strengths.csv` | Output of the script above. Also copied to `/home/alan/Documents/multi_2/Attachments/`. |
| `knickebein_paths.csv` | Input path dataset: station-target pairs with DEM-verified TX altitudes, operational reach, ground type, source citations. |
| `graphs/*.png` / `graphs/*.gif` | Plot outputs. PNGs are also copied to the vault Attachments folder. |

## Run order

```
cd Knickebein/
python3 compute_signal_strengths.py            # writes the canonical CSV
python3 make_p526_vs_p368_graphs.py            # 13 bars + sweep PNGs
python3 knickebein_beam_map.py                 # operational coverage map
python3 knickebein_beam_map_telefunken.py      # 5 Telefunken-variant PNGs (loops internally)
```

The first script does not depend on the others. The `make_p526` and `knickebein_beam_map` scripts import the library directly rather than re-reading the CSV, but the CSV is the canonical number set for any downstream tool. `knickebein_beam_map_telefunken.py` DOES read the CSV for its bar-chart inset.

The two beam-map scripts need `/tmp/countries.geojson` (not tracked here; pulled from the public `datasets/geo-countries` dataset on demand — see README).

## Canonical parameters (set in `botb_itu_analysis.py`)

| Constant | Value | Notes |
|---|---|---|
| `FREQ_HZ` | 31.5e6 | Knickebein Kn-4 / Kn-2 operational frequency, EBL 3 channel 16 |
| `TX_POWER_W` | 3000 | 3 kW, "Knickebein Large" 99 m x 29 m |
| `RX_GAIN_DBI` | 0.0 (or 3.0 in SN path) | Isotropic baseline |
| `RX_BW_Hz` | 500 | MCW / A2 matched-filter bandwidth. See correction #1 below. |
| `N_FLOOR_dBW` | -159.0 (derived) | Galactic + thermal per ITU-R P.372, with the 500 Hz BW |
| `CROSSOVER_dB` | -19.87 (derived) | Equisignal crossover loss at 5 deg squint; `20·log10|sinc(L·sin(5°)/λ)|` |
| `SNR_TO_DBUV` | -22.0 (derived) | = N_FLOOR_dBW + 137, converts SNR dB to μV at 50 ohm |
| `V_NOISE_UV` | 0.0795 | Noise floor in μV at 50 ohm |
| `EQUISIGNAL_HALF_ANGLE_DEG` | 0.0277 (derived) | 1 dB pilot A/N threshold on sinc-slope at 5° squint — full corridor ≈ 0.055° |

## Corrections applied April 2026

### 1. Detection bandwidth 3000 Hz → 500 Hz

The emission mode is Morse-style keyed CW (MCW / ITU A2), not AM voice. The correct detection bandwidth is a matched filter on the Morse symbol rate, not 3 kHz audio. `bullnyte` caught this on Discord. Everything below follows.

- Noise floor shifted from `-151.2 dBW` to `-159.0 dBW` (7.8 dB improvement)
- Reference voltage shifted from `0.19 μV` to `0.0795 μV`
- Every SNR result moves `+7.8 dB` across the board
- Classification: Kleve → Birmingham moves MARGINAL → USABLE. Every Stollberg → Midlands path stays DEAD. Null verdict unchanged.

### 2. `SNR_TO_DBUV` made dynamic everywhere

Used to be hard-coded. Now derived from `N_FLOOR_dBW + 137.0` in both graph scripts and the beam map. If the bandwidth ever changes again, you only edit the library and every consumer updates.

### 3. Peak → equisignal applied uniformly on graphs

The Friis / Sommerfeld-Norton / Fock functions return the **peak** (on-boresight) SNR. Knickebein navigation happens on the equisignal between the two lobes, which is 19 dB below peak at the 5 deg Knickebein squint. The graph scripts used to plot peak but label the axis as equisignal. Fixed by applying `+ CROSSOVER_dB` at every plotting site and relabeling titles to "equisignal V_rx at 50 ohm (5 deg squint)".

### 4. β fix for vertical polarisation over sea (pre-existing, do not re-edit)

`itu_diffraction_loss()` already applies the P.526-16 Eq. 16/16a correction. If you see numbers that look ~10 dB too high over sea paths, check this first before changing anything.

### 5. Telefunken-variant visual sweep (this rewrite)

`knickebein_beam_map_telefunken.py` now takes a `HIGHLIGHT_KM` parameter and renders one map per Telefunken target (1000 / 800 / 700 / 500 / 400 km) in a single run:

- Stollberg beam aims at the highlighted target; its total projected length is held constant (≈12.25 units ≈ 1360 visual km) across all five images so the sequence looks consistent when animated.
- The highlighted target gets a cyan glow halo + glowing tag + bar-chart row backdrop to indicate which one is under test.
- Kleve beam and the green/pink FE-vs-Ge strength wedges stay fixed (they represent Kleve's geometry, not Stollberg's).
- GIF output (`knickebein_beam_map_telefunken_cycle*.gif`) is built separately via ffmpeg — see README for the command.

### 6a. Equisignal corridor width — first-principles library function

`botb_itu_analysis.py` now exposes `equisignal_corridor_width_m(distance_m)` and the helpers `sinc_pattern_slope_dB_per_rad()` / `equisignal_half_angle_rad()`. These replace the previous hardcoded `THETA_EQ_DEG = 0.066°` in `compute_equisignal_widths.py` with a derivation from the same sinc aperture pattern that gives `CROSSOVER_dB`, combined with a 1 dB pilot A/N discrimination threshold (NATO AGARDograph 300 Vol. 10 §6.2). At 439 km (Bufton Spalding, 21 Jun 1940) the library predicts ≈ 424 m / 464 yd — 8 % under Bufton's visual estimate of 400-500 yd, inside measurement tolerance. The Python and spreadsheet now agree on the full width at every path. `CROSSOVER_dB` is derived the same way and evaluates to -19.87 dB (was previously rounded to -19 in-line).

### 6b. Visual-position interpolation on the Kleve LoS

`_kleve_visual_frac()` (inside the telefunken map) does piecewise-linear interpolation between the fiducial `VISUAL_FRACS` marker positions. Raw `d_km / VISUAL_BEAM_SPAN_KM` under-placed the Fock crossover (~600 km real) before the 500 km marker. Interpolation puts it visually between the 500 and 700 markers, which is where an observer expects 600 km to land.

## What we are still hunting for

These gaps would let us tighten the null further. We have not found primary-source numbers yet.

### Dot / dash keying rate in Hz

- Why it matters: sets the theoretical minimum matched-filter bandwidth
- What we have: `500 Hz` is a conservative upper bound; the Morse symbol rate for the Knickebein beacon was probably much lower
- Would lower the noise floor further if we confirm a slower rate

### EBL 3 receiver sensitivity in μV at the antenna terminal

- Why it matters: spot-check our derived `-159 dBW` against the manufacturer's own spec
- What we have: only the operational spec "70 km at 200 m altitude, 500 W TX" from D.(Luft) T.4058 §21
- No μV figure quoted anywhere in the service manual

### Siebkreis SK 3 dB bandwidth in Hz

- Why it matters: the actual EBL 3 audio detection bandwidth is set by the two-pole LC filter in the anode of tube (47) in the schematic, tuned to 1150 Hz with 700 Hz and 1700 Hz rejection. Design constraint implies ~60 to 120 Hz.
- What we have: verbal rejection spec from D.(Luft) T.4058 p. 17 ("zusammenbrechen" at 700 and 1700), but no numeric bandwidth
- Would lower the noise floor by roughly another 7 dB if we confirm ~100 Hz

## Where the outputs are consumed

- Null doc: `/home/alan/Documents/multi_2/Null_Hypothesis/Battle_of_the_Beams/Knickebein_Propagation_Null.md`
- Graphs reference: `/home/alan/Documents/multi_2/Null_Hypothesis/Battle_of_the_Beams/GRWAVE_P368_BotB.md`
- Falsification replies: `/home/alan/Documents/multi_2/Null_Hypothesis/Battle_of_the_Beams/Falsification_Attempts/DC_Dan.md`
- Vault copies of CSV and PNGs land in `/home/alan/Documents/multi_2/Attachments/`

## If a downstream number disagrees with the CSV

The CSV is the source of truth. If a graph, a map, or a null-doc cell disagrees, regenerate it from `compute_signal_strengths.py` first before editing any prose.
