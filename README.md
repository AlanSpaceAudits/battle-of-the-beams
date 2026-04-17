# Battle of the Beams — spherical-Earth null hypothesis

Physics and source-archive repo for a null-hypothesis test of three WWII German VHF beam-bombing systems — **Knickebein**, **X-Gerät**, and **Y-Gerät** — against ITU-R P.526-16 Fock diffraction on a 6,371 km spherical Earth with the standard 4/3 refractivity model.

The Knickebein analysis is the load-bearing test. The X-Gerät / Y-Gerät subdirectories hold companion path datasets and per-target bar charts; their station geometry is French-coast with operational ranges inside the LoS horizon, so they don't discriminate globe vs flat and H₀ survives by default.

## Current verdict (Knickebein)

Of the 9 confirmed Kleve + Stollberg operational paths, only **Kleve → Spalding** (the shortest, where Bufton physically measured the beam in June 1940) passes the +10 dB detection floor on the globe. Every Stollberg path is 30–65 dB below the galactic noise floor. Of the 6 Telefunken July 1939 over-sea range tests, only the 400 km configuration passes; 700 / 800 / 1000 km fail by 31–97 dB. Flat-Earth Friis and Sommerfeld-Norton plane-Earth both show +40–50 dB of equisignal headroom at every path.

Eckersley told British intelligence this in June 1940 and was overruled by Churchill on circumstantial evidence. The documented operational history of Knickebein guiding He 111s to Derby / Birmingham / Liverpool for ten months is inconsistent with spherical-Earth VHF propagation at 31.5 MHz — by 35–66 dB on the primary targets.

## Repo layout

```
BotB/
├── botb_itu_analysis.py              shared physics library
├── grwave/                           vendored grwave package (P.368 cross-check)
├── sources/                          primary-source PDFs, HTML snapshots, OCRs
├── Knickebein/                       main analysis — see Knickebein/SCRIPTS.md
│   ├── knickebein_paths.csv
│   ├── botb_signal_strengths.csv     canonical numbers, regenerable
│   ├── compute_signal_strengths.py
│   ├── make_p526_vs_p368_graphs.py
│   ├── knickebein_beam_map.py
│   ├── knickebein_beam_map_telefunken.py
│   └── graphs/                       PNGs + GIFs
├── XGerat/
│   ├── xgerat_paths.csv
│   └── graphs/                       peak + equisignal bar charts per target
└── YGerat/
    ├── ygerat_paths.csv
    └── graphs/                       peak + equisignal bar charts per target
```

## Running the Knickebein pipeline

```bash
cd Knickebein/
python3 compute_signal_strengths.py            # writes botb_signal_strengths.csv
python3 make_p526_vs_p368_graphs.py            # 13 bar-chart + sweep PNGs
python3 knickebein_beam_map.py                 # operational beam-coverage map
python3 knickebein_beam_map_telefunken.py      # 5 Telefunken-target maps
```

Outputs land in `Knickebein/graphs/`. Canonical numbers also get copied to `/home/alan/Documents/multi_2/Attachments/` for the Obsidian vault.

### Dependency: `countries.geojson`

The two beam-map scripts read `/tmp/countries.geojson` for coastline outlines. If it's missing, pull the standard public dataset once:

```bash
curl -sSL -o /tmp/countries.geojson \
  https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson
```

### Animated Telefunken cycle (GIF)

`knickebein_beam_map_telefunken.py` writes five PNGs — one per highlighted target. The looping GIFs in `Knickebein/graphs/` were built from those PNGs with ffmpeg:

```bash
# 3 s/frame, looping 1000 → 800 → 700 → 500 → 400 km (scaled 1280 px)
ffmpeg -y -f concat -safe 0 -i frames.txt \
  -vf "fps=10,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=full[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  -loop 0 knickebein_beam_map_telefunken_cycle.gif
```

See `Knickebein/SCRIPTS.md` for details and the 1 s / full-res variants.

## Canonical parameters (set once in `botb_itu_analysis.py`)

| Constant | Value | Notes |
|---|---|---|
| `FREQ_HZ` | 31.5e6 | Knickebein Kn-4 / Kn-2 operational frequency, EBL 3 channel 16 |
| `TX_POWER_W` | 3000 | 3 kW, "Knickebein Large" 99 m × 29 m aperture |
| `G_DIR_dB` | 26.0 | TX directivity, `4πA/λ²` |
| `RX_GAIN_DBI` | 0.0 | isotropic baseline (Sommerfeld-Norton path uses 3.0) |
| `RX_BW_Hz` | 500 | MCW / ITU A2 matched-filter bandwidth |
| `N_FLOOR_dBW` | −159.0 | galactic + thermal, ITU-R P.372 Eq. 14 `Fa = 52 − 23·log₁₀(f_MHz)` at 500 Hz BW |
| `V_NOISE_UV` | 0.0795 | noise floor in μV at 50 Ω |
| `CROSSOVER_dB` | −19.0 | equisignal crossover loss at 5° squint |
| `DETECT_dB` | +10.0 | bare RF detection threshold above noise floor |

**Effective Earth radius:** 8,495 km (k = 4/3, ITU-R P.453 standard atmosphere).
**TX heights:** Kleve 111 m (83 m terrain + 28 m Trenkle frame), Stollberg 72 m (44 m + 28 m).
**Aircraft altitude:** 6,000 m (He 111 operational), 4,000 m (Telefunken July 1939 range tests).

## Change log (April 2026 rewrite)

### Detection bandwidth 3000 Hz → 500 Hz

The emission mode is Morse-style keyed CW (MCW / ITU A2), not AM voice. The correct detection bandwidth is a matched filter on the Morse symbol rate, not 3 kHz audio. Caught by `bullnyte` on Discord. Consequences:

- Noise floor: −151.2 dBW → **−159.0 dBW** (7.8 dB improvement)
- Reference voltage: 0.19 μV → **0.0795 μV**
- Every SNR moves uniformly +7.8 dB
- Classification boundaries unchanged: Kleve → Birmingham moves MARGINAL → USABLE; every Stollberg → Midlands path stays DEAD. Null verdict unchanged.

### `SNR_TO_DBUV` made dynamic

Was hard-coded; now derived as `N_FLOOR_dBW + 137.0` in every consumer. Edit the library once and every graph/map updates.

### Peak → equisignal applied uniformly on graphs

The Friis / Sommerfeld-Norton / Fock functions return **peak** (on-boresight) SNR. Knickebein navigation happens on the equisignal between the two lobes, which is 19 dB below peak at the 5° Knickebein squint. Graph scripts used to plot peak values under an "equisignal" axis label. Fixed by applying `+ CROSSOVER_dB` at every plot site.

### β correction for vertical polarisation over sea

`itu_diffraction_loss()` applies the P.526-16 Eq. 16 / 16a correction for v-pol at f < 300 MHz over sea:

- `K² ≈ 6.89·σ / (k^(2/3)·f^(5/3))`
- `β = (1 + 1.6K² + 0.67K⁴) / (1 + 4.5K² + 1.53K⁴)`
- At σ = 5 S/m (sea), f = 31.5 MHz, k = 4/3: β ≈ **0.810**

Also implemented the Eq. 18 G(Y) lower-bound clamp `G(Y) ≥ 2 + 20·log₁₀(K)`. Made Stollberg and Telefunken sea paths 17–27 dB stronger than pre-β values; operational verdict unchanged.

### Sommerfeld-Norton plane-Earth curve added

`sommerfeld_norton_Ez()` and `sommerfeld_norton_snr_peak()` implement the ITU Handbook on Ground Wave Propagation (2014) Part 1 §3.2.1 three-term form: direct ray + Fresnel-reflected ray + Norton 1937 surface-wave attenuation function `F` (via `scipy.special.wofz` with large-|w| asymptotic fallback). Plotted as the gold-yellow envelope alongside Friis.

### Telefunken map refactor

`knickebein_beam_map_telefunken.py` was parametrised over `HIGHLIGHT_KM`. A single run emits five PNGs — one per Telefunken target (1000 / 800 / 700 / 500 / 400 km) — with the Stollberg beam realigned to each target. Total Stollberg beam length is held constant across the five images so the sequence animates cleanly. The highlighted target gets a cyan glow halo and bar-chart row backdrop. Inline `_kleve_visual_frac()` piecewise-linear interpolation keeps the Fock-crossover wedge lined up visually between the 500 and 700 km markers (matching observer expectation for the ~600 km crossover).

## Primary sources

All numerical claims trace to a file in `sources/` or an ITU recommendation. Key entries:

- `1939_BArch_RL19-6-40_230Q7_Nutzbereich.md` and the companion Telefunken range-test appendix (230Q8 App. 2)
- `D-Luft-T-4058-FuBl-2-Geraete-Handbuch-1943.pdf` — receiver service manual (EBL 3 / FuBl 2, AGC + ALC + 2-stage AF architecture; settles the +10 dB bare-detection threshold)
- `ARC_RM_2296_Cameron_Annand_1942_bomber_cabin_noise.pdf` — cockpit SPL measurements
- `Bauer_Navigati.pdf` — Bauer (2004) secondary summary
- Dörenberg HTML snapshots — Knickebein station database (Kn-1 through Kn-13)
- `Radiomuseum_EBL3.html`, `Radiomuseum_EO509.html` — receiver provenance

See `sources/README_primary_sources_digest.md` for the full index.

## Vault cross-references

These docs live in the Obsidian vault and consume the outputs of this repo:

- `Null_Hypothesis/Battle_of_the_Beams/Knickebein_Propagation_Null.md` — primary null-hypothesis analysis
- `Null_Hypothesis/Battle_of_the_Beams/XGerat_Propagation_Null.md` — X-Gerät + Y-Gerät companion
- `Null_Hypothesis/Battle_of_the_Beams/Prezzie_BotB.md` — presentation notes
- `Null_Hypothesis/Battle_of_the_Beams/Falsification_Attempts/DC_Dan.md` — falsification replies

PNG attachments and the CSV are mirrored to `multi_2/Attachments/`.

## If a downstream number disagrees with the CSV

The CSV is the source of truth. Regenerate it from `compute_signal_strengths.py` first before editing any prose or plot.
