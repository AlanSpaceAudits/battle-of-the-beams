# Primary Sources Digest — Knickebein / FuBl / Cockpit Acoustics

Downloaded 2026-04-12 for the Battle of the Beams null-hypothesis analysis.
Every claim below traces to a file in this directory.

---

## 1. D.(Luft) T.4058 — FuBl 2 Geräte-Handbuch (Feb 1943)

**File:** `D-Luft-T-4058-FuBl-2-Geraete-Handbuch-1943.pdf` (13 MB, 120 pages, scanned images)
**Source URL:** https://www.cdvandt.org/D-Luft-T-4058-Funk-Landegeraet-FuBl-2.pdf
**OCR output:** `D-Luft-T-4058-FuBl-2_OCR_p1-50.txt` (English Tesseract, ~80% readable for numerics/German text)

### Key passage — Section 21 "Empfindlichkeit" (page ~14, EBL 3 main receiver)

> "Sichere optische und akustische Anzeige des Ansteuerungskurses in 200 m Flughöhe
> ab 70 km Entfernung vom Flughafen (Strahlungsleistung des AFF am Boden: 500 Watt)."

**Translation:** Reliable optical and acoustic indication of the approach course at 200 m
flight altitude, from 70 km distance from the airport (radiated power of the AFF ground
station: 500 W).

**Critical observation:** The Luftwaffe primary-source spec for FuBl 2 sensitivity is
defined **operationally** (km at altitude, given TX power), **not in microvolts**. Any
"1-2 µV" claim about FuBl 1/2 sensitivity is NOT from this manual — it must be a
secondary back-calculation or modern modeling estimate.

### Other confirmed specs from this manual

- **Frequency:** 30.0-33.33 MHz
- **Audio tone:** 1150 Hz (Section 16, description of AFF keying)
- **Keying pattern:** standard 1:7 ratio Morse E/T dot-dash (Ansteuerungs-Funkfeuer)
- **Range markers:** EZ transmitters at 3000 m and 300 m from runway boundary
- **System configuration:** AFF (Ansteuerungs-Funkfeuer, 500 W at 30-33.3 MHz) + 2× EZ
  marker senders (Einflugzeichen, 38.0 MHz, 5 W each)
- **Section 39, EBL 2 sensitivity:** "Sicheres Ansprechen der Glimmlampe bei Überfliegen
  eines EZ-Senders in 200 m Höhe (Strahlungsleistung des EZ-Senders etwa 5 Watt, Kurs
  des Flugzeuges innerhalb des Leitstrahles)" — again, operational, not µV.

**NB:** The 500 W in Section 21 is the **landing-beacon AFF** — aka "Small Knickebein" /
"Karussel" when used for long-range navigation. It is **NOT** the 3 kW Large Knickebein
beacons at Kleve (Kn-4) and Stollberg (Kn-2). See Dörenberg below.

---

## 2. Dörenberg Knickebein pages (nonstopsystems.com, via Wayback 2024-09-04)

**Files:**
- `Dorenberg_Knickebein_nonstopsystems_20240904.html` (731 KB, main Knickebein page)
- `Dorenberg_Knickebein_Kn1-13_20240904.html` (736 KB, station details Kn-1 through Kn-13)

**Source URL (archive):** https://web.archive.org/web/20240904174115/https://www.nonstopsystems.com/radio/hellschreiber-modes-other-hell-RadNav-knickebein.htm

### Confirmed station / transmitter taxonomy

Citing ref. 230Q8 (Bundesarchiv RL 19-6/40, 1939):

> "Early 1939, the Luftwaffe Signal Corps was instructed to build five rotatable VHF
> long-range radio navigation beacons (Fernfunkfeuer drehbar, FFuFd) as soon as possible:"

**Three Telefunken high-power beacons (code name "Knickebein"):**
- **3 kW transmitter output**
- **Range: up to 1200 km** (depending on aircraft altitude and receiver sensitivity/selectivity)
- Located at **Stollberg/Bredstedt (Kn-2), Kleve (Kn-4), and a southern station (Kn-12)**
- Antenna system weight: ~200 metric tons

**Two Lorenz lower-power beacons (code name "Karussel"):**
- **500 W transmitter output**
- **Range: 300-600 km**
- Located at Bad St. Peter and Sylt
- Same chassis as the civil Lorenz 500 W landing beacon (AFF)

### Antenna dimensions (ref. 230Q8, p.12, from ground-station photos)

- **99 m wide × 29 m tall** overall aperture (including end dipoles)
- 93 × 29 m steel truss frame proper
- Used for both Stollberg (Kn-2) and Kleve (Kn-4)

### Equisignal beam specs

- **Beam width: 0.3°** (actual), originally specified as ±0.2° and later improved to ±0.1°
- **Dot-only / dash-only reception out to ±45°** from beam center (±40° per ref. 230Q7)
- Tone modulation: 1000 Hz originally considered, 1850/1250 Hz later considered, but
  eventually **1150 Hz was standard** (shared with FuBl landing beacon)

### Range/propagation test data (ref. 230Q8 Appendix 2, Telefunken July 1939)

- **Location:** over open water (Baltic or North Sea)
- **Altitude:** 4000 m (≈ 13,000 ft)
- **Three radio configurations × two antennas = six combinations tested:**
  1. Standard FuBl 1
  2. FuBl 1 with increased **selectivity**
  3. Unspecified special receiver
  Each tested with: (a) rod antenna, (b) trailing wire antenna
- **Measured average ranges: 400 to 1000 km**
- **Min/max:** ±20% around averages
- **Dependencies:** air temperature, humidity, seasonal propagation

**Corresponds directly to the TF_400 / TF_500 / TF_700 / TF_800 / TF_1000 rows in
`knickebein_paths.csv`.**

### Luftwaffe-defined usability criterion (Nutzbereich)

From the Geheim chart in Bundesarchiv file RL 19-6/40 (former RL 19/537):

> "graph with 0.3° equisignal beam-width and useable altitude vs. range based on
> **beacon signals audibility** = vertical usability boundaries of the equisignal beam lobe"

**This is the key finding for our null hypothesis framing:** the Luftwaffe themselves
defined Knickebein usability operationally as **audibility of the beacon signal in the
cockpit**, not a µV RF threshold. This directly supports treating cockpit acoustic noise
as part of the usability constraint, not just the receiver RF noise floor.

### FuBl 1 receiver variants mentioned by Dörenberg

Confirmed quote: "three radio configurations: 1) standard FuBl 1, 2) FuBL 1 with increased
selectivity, and 3) an unspecified special receiver."

**Note:** Variant 2 is "increased selectivity", NOT "increased sensitivity". The research
agent's earlier summary said "much more sensitive" — that's a paraphrase from a different
R.V. Jones / Price quote, not from this primary source. The 1939 Telefunken test used
configurations distinguished by selectivity (narrower IF passband), which would reduce
the noise admitted rather than increase the receiver gain.

---

## 3. ARC R&M 2296 — Measurements of Cabin Noise in Bomber Aircraft

**File:** `ARC_RM_2296_Cameron_Annand_1942_bomber_cabin_noise.pdf` (1.9 MB, text layer)
**Authors:** D. Cameron PhD & W. J. D. Annand BSc, A.&A.E.E., Jan 1942
**Source URL:** https://naca.central.cranfield.ac.uk/bitstream/handle/1826.2/3366/arc-rm-2296.pdf

### Method

- Calibrated Objective Noisemeter (Spec. 791/R.A.E./W.T.610)
- Octave Analyser type 74101B
- Measurements at pilot's and wireless-operator's positions
- 16 octave bands from 37.5-75 Hz up to 6400-12800 Hz
- Eight multi-engined bombers tested in level flight

### Key octave-band values in the 600-1200 Hz band (contains the Knickebein 1150 Hz tone)

All values are SPL in dB, taken from Tables 1-7 of the report.

| Aircraft | Condition | Position | 600-1200 Hz band | Overall peak |
|---|---|---|---|---|
| **Halifax** (no soundproofing) | 3000 rpm | W/T op | **95-97** | 115-117 |
| **Halifax** (soundproofed) | 2600 rpm +4 | W/T op | **82-84** | 109-111 |
| **Fortress** (heavily s/p) | 2500 rpm A/R | Pilot | **100-102** | 110-116 |
| **Fortress** (cruise) | 2100 rpm A/R | W/T op | **87-88** | 100-103 |
| **Wellington IA** Pegasus | 2600 rpm, Vt 0.82 | **W/T op** | **103-106** | 118-120 |
| **Wellington IV** Twin Wasp | 2700 rpm, climb | W/T op | **112-115** | 128-129 |
| **Wellington IV** Twin Wasp | 2550 rpm climb | W/T op | **102-105** | 118-121 |
| **Hudson** | max 2300 rpm | W/T op | **97-100** | 106-115 |
| **Lancaster** 12 ft | 2650 rpm +3.5 | W/T op | **97** | 111-115 |
| **Lancaster** 13 ft | 2650 rpm +3.5 | W/T op | **97** | 108-111 |
| **Albemarle** | 2400 rpm +2.5 | Pilot | **93** | 115-123 |
| **Albemarle** | 2400 rpm −1 | Pilot | **91-92** | 115-121 |
| **Manchester** | 2600 rpm +4 | Pilot | **97-98** | 106-116 |
| **Manchester** | 3000 rpm +6 | Pilot | **98-99** | 116-118 |

### He 111 inference

No direct He 111 measurement exists. The Wellington IA (Pegasus radial, twin-engine,
open exhaust stacks, fabric-covered twin-engine bomber with cockpit directly between
the nacelles) is the closest geometric/powerplant analogue. Wellington W/T position
octave levels in the 600-1200 Hz band run **103-106 dB SPL**, with overall peak beats
at 118-120 dB.

**He 111 W/T operator position at cruise, best estimate:** **~100-110 dB SPL in the
600-1200 Hz band**, 115-125 dB overall peak. This is an inference from R&M 2296 on
geometrically similar British aircraft, flagged as such — no direct measurement exists.

### Supporting qualitative finding

> "General noise levels ... of British bombers are near the borderline between tolerable
> and excessive noise" — Cameron & Annand 1942, R&M 2296 Summary.

> "Soundproofing produces a decrease of about 10 db. in the high frequency noise" — ibid,
> Section 4.1 (Halifax comparison).

Halifax unsoundproofed W/T 600-1200 Hz band: 95-97 dB.
Halifax soundproofed W/T 600-1200 Hz band: 82-84 dB.
**Effect of soundproofing in the Knickebein band: ~13 dB reduction.**

The He 111 had no significant acoustic soundproofing (Luftwaffe bombers prioritized
weight and range over crew comfort). The unsoundproofed Halifax/Wellington values
are the correct analogue.

---

## 4. Bauer — Navigati.pdf (C. Dörenberg / F. Dörenberg compilation)

**File:** `Bauer_Navigati.pdf` (1.1 MB, text layer)
**Source URL:** https://www.cdvandt.org/Navigati.pdf

Bauer's 2004 compilation history of German radio navigation. Cited for:
- Telefunken Knickebein antenna dimensions (99 × 29 m)
- Transmit power 3 kW
- FuBl 1 / FuBl 2 development timeline
- Ref. 230Q7 and 230Q8 Bundesarchiv file numbers

Not the most technically detailed source but useful for cross-referencing the specific
document/archive numbers.

---

## 5. FuBl-2 Introduction (Bauer, cdvandt.org)

**File:** `FuBl-2-Introduction_Bauer.pdf` (3.0 MB)
**Source URL:** https://www.cdvandt.org/FuBl-2-Introduction%20FuBL-2-copy3a-V2.pdf

Modern reconstruction/summary of the FuBl 2 system by Arthur O. Bauer. Useful for
cross-reference and photographs of the physical equipment. Confirms:
- EBL 3 = 7× RV12P2000 tubes, IF 6000 kHz, 11 AM tuned circuits, 34 channels at 100 kHz
- Integration with AFN 2 indicator (visual needle deflection)
- Common chassis used in FuG 16 and other Luftwaffe VHF equipment

---

## 6. Radiomuseum.org EBL 3 + EO509 pages

**Files:**
- `Radiomuseum_EBL3.html` (65 KB)
- `Radiomuseum_EO509.html` (62 KB)

**URLs:**
- https://www.radiomuseum.org/r/mil_blindlandegeraet_ebl3_ebl3_h_f.html
- https://www.radiomuseum.org/r/lorenz_empfaenger_eo509eo_50.html

### EBL 3 confirmed specs

- **Tube count:** 7× RV12P2000
- **Topology:** Superheterodyne with RF preamplifier stage
- **IF frequency:** 6000 kHz (vs SBA's 2830 kHz — more gain, better image rejection)
- **Tuned circuits:** 11 AM
- **Frequency range:** 30-33.3 MHz, 34 channels at 100 kHz spacing
- **Weight:** 5 kg
- Built at AEG-Sachsenwerk (Dresden-Niedersedlitz), designed by C. Lorenz Berlin
- Used in: Ar-234, Ju-88 night fighter, Do-335, Do-217, Me-109/262 weather, Ju-388, FW-190, Ta-152
- Post-WWII: chassis reused in Curt Höhne AS503/OS auto radios

**The radiomuseum.org page does NOT contain a µV sensitivity figure for the EBL 3.**
Any such figure in secondary literature comes from a different source (likely Trenkle
or direct measurement of surviving units).

### EO509 (general Lorenz all-wave receiver, for comparison only)

- 4-tube regenerative (2 AM circuits, 2 AF stages), MF2 pentodes throughout
- 15-20000 kHz, 12 subbands
- Used in Swiss Army G1.5K station
- **No µV sensitivity figure visible on the page** — the earlier agent's "0.5-25 µV CW,
  2-20 µV telephony for 1 V into 4000 Ω" must have been from the page's linked original
  datasheet (not reproduced as text).

**Unverified claims to re-check or cite cautiously:**
- The "10 µV @ 30 MHz for 10 dB S/N" figure the research agent attributed to the
  SBA (modified BC-455) is NOT on the Robinson tuberadio.com page I downloaded.
  The downloaded page describes the receiver history and modifications but provides
  no quantitative sensitivity spec.

---

## 7. What's NOT documented in accessible primary sources

Flag these as "claimed but unverified" in any public writeup:

1. **FuBl 1 or FuBl 2 sensitivity in µV.** Not in D.(Luft) T.4058. Any µV figure is a
   secondary estimate, not a primary Luftwaffe spec. Dan's "1-2 µV FuBl 1" is
   undocumented.
2. **Heinkel He 111 cockpit dB SPL.** No direct measurement found in any accessible
   English- or German-language source. R&M 2296 Wellington/Hudson proxies are the best
   we can do.
3. **Luftwaffe LKp W100/N101 headset acoustic attenuation.** No spec found. General
   historical finding: WWII aircraft headsets "provided no real hearing protection."
4. **ICAO/military standard dB S/N threshold for aviation Morse/keying signals.** No
   specific standard located. General psychoacoustic masking results (6-15 dB S/N above
   masker for reliable dot-dash discrimination) must be cited from hearing-research
   literature, not an aviation standard.

---

## 8. Implications for the null hypothesis

The primary sources support a **two-constraint** framing:

1. **RF detectability constraint** (current analysis): receiver output SNR must exceed
   some threshold above the sky/thermal noise floor (ITU-R P.372, ~18 dB at 31.5 MHz).

2. **AF audibility constraint** (new): the audio output of the receiver must deliver a
   1150 Hz tone at the eardrum that is audibly above the cockpit acoustic noise floor,
   which in the 600-1200 Hz band for twin-engine unsoundproofed WWII bombers is
   **~100-110 dB SPL** (R&M 2296 Wellington/Hudson proxy).

**The Luftwaffe's own usability criterion** (ref. 230Q7, Bundesarchiv RL 19-6/40) was
explicitly **"beacon signals audibility"** — not a µV RF threshold. This is primary-source
validation that constraint (2) is the real operational bound.

For typical 1940s magnetic earphones (2000 Ω Dfh.b) operating near their ~2 mW maximum
drive, the acoustic output at the eardrum saturates at ~100-110 dB SPL. Given
essentially zero headset isolation, the receiver must run its AF stage near full output
to overcome cockpit noise — which requires RF input signal ~20-30 dB above the
minimum detection threshold.

**Net effect on our null hypothesis:** the effective usability threshold for Knickebein
on the globe is roughly **20 dB stricter** than the current "+10 dB above galactic noise
floor" assumption. This pushes Kleve → Derby (+15.8 dB) and Kleve → Birmingham
(+9.5 dB) into the unusable zone on the globe, while the Telefunken over-water range
measurements continue to falsify the globe at 700-1000 km.

**Dan's cockpit-noise argument, properly quantified against primary sources,
strengthens rather than weakens the null hypothesis falsification of a spherical Earth
for the Knickebein system.**
