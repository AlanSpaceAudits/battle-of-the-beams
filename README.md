# Battle of the Beams: VHF Propagation Analysis

Quantitative analysis of the WWII Knickebein beam navigation system, comparing predictions from rectilinear (flat) propagation against spherical-Earth (Fock diffraction) models.

## What is Battle of the Beams?

During WWII, Germany used the Knickebein system to guide bombers to British targets at night. Two narrow VHF beams (31.5 MHz) were transmitted from continental Europe. A pilot flew along one beam and released bombs when a second cross-beam was detected. The beams used the Lorenz equisignal technique: two slightly offset sub-beams alternate dots and dashes, and the pilot stays on course where both are equal in amplitude.

British monitoring at Spalding measured the equisignal corridor at 400-500 yards wide, at a range of 440 km from Kleve, Germany.

## What this repo computes

### `botb_propagation.py` — Main analysis

Computes for each path (Kleve to Spalding, 440 km; Stollberg to Beeston, 694 km):

- **Geometric analysis**: Earth curvature drop, radio horizon distances, shadow zone lengths
- **Flat-surface predictions**: Beam widths from antenna Fourier optics, equisignal width from measured divergence angle
- **Diffraction analysis**: Creeping-wave ribbon width, knife-edge beam width
- **Fock smooth-Earth diffraction loss**: 5-mode residue series (Sommerfeld-Watson transformation)
- **Equisignal crossover correction**: SNR at the actual equisignal crossover point, not the beam peak
- **Squint/SNR trade-off table**: Shows that on a globe, no squint angle gives both a narrow equisignal and a detectable signal
- **Link budget**: Received power vs. noise floor (galactic noise per ITU-R P.372)

### `equisignal_prediction.py` — Equisignal geometry

Predicts the equisignal corridor width from first principles using:

- The sinc beam pattern from the 99 m aperture (Fraunhofer diffraction)
- The Lorenz beam-splitting geometry (two squinted sub-beams)
- Receiver detection threshold (instrument JND + noise-limited sensitivity)
- Parametric scan over squint angles

### `equisignal_globe_honest.py` — Step-by-step globe model walkthrough

The main analysis (`botb_propagation.py`) gives the globe model its best shot: peak SNR, clean beam pattern assumptions. This script asks: even with those generous numbers, what does the signal *actually look like* after diffracting around a sphere?

It walks through 6 steps of what happens to the beam on a globe:

1. **Beam leaves the antenna** -- pattern intact, equisignal structure preserved
2. **Beam reaches the horizon** (58 km) -- still coherent, no issues yet
3. **Beam couples into creeping waves** -- vertical pattern destroyed, energy smeared into a 47.8 km ribbon regardless of original beam geometry
4. **Creeping wave propagates in shadow zone** -- exponential decay at 0.292 dB/km per mode, multiple modes with different phase velocities
5. **What the aircraft actually receives** -- SNR, amplitude fluctuations, Rayleigh fading statistics, AGC tracking failure during signal dropouts
6. **Equisignal coherence analysis** -- inter-modal dispersion, comparison noise from amplitude instability, whether the Lorenz meter can still discriminate dots from dashes

The name "globe_honest" is the point: the main script computes the numbers; this script examines whether the *quality* of the signal (not just its strength) could support precision navigation. Even if you had marginal SNR at beam peak, the creeping wave regime produces a fading, dispersed, multipath signal that a Lorenz equisignal receiver cannot use.

## Key results

| Quantity | Flat Model | Globe Model | Measured |
|---|---|---|---|
| Equisignal width (Kleve, 440 km) | 554 yd | N/A | **500 yd** |
| SNR at equisignal (Kleve) | **66.7 dB** | **-10.4 dB** | sufficient |
| SNR at equisignal (Stollberg) | **62.7 dB** | **-86.5 dB** | sufficient |

On a globe, the equisignal crossover loss (-19 dB) combines with the Fock diffraction loss (-77 dB for Kleve, -149 dB for Stollberg) to place each individual dot/dash signal below the noise floor. No squint angle simultaneously produces a narrow equisignal and a detectable signal.

On a flat surface, the 86 dB of SNR headroom absorbs any crossover depth.

## Dependencies

- Python 3.6+
- NumPy
- SciPy (for `scipy.special.airy` — Airy function in the Fock residue series)

## Usage

```bash
python3 botb_propagation.py        # Full analysis with tables
python3 equisignal_prediction.py   # Equisignal geometry scan
python3 equisignal_globe_honest.py # Step-by-step globe walkthrough
```

Each script prints formatted output to the terminal and saves it to a corresponding `_output.txt` file.

## References

1. Fock, V.A. (1965). *Electromagnetic Diffraction and Propagation Problems*. Pergamon Press, Oxford. Ch. 10 reprints Fock (1945). Residue series: Eq. (6.10), p. 209. Flat Earth (Weyl-van der Pol): Eq. (3.23), p. 201. Equivalent radius critique: Ch. 13, p. 254.
2. Eckersley, T.L. (1937). "Ultra-Short-Wave Refraction and Diffraction." *J.I.E.E.* 80, p.286.
3. Vogler, L.E. (1961). "Smooth Earth Diffraction Calculations for Horizontal Polarization." *NBS J. Res.* 65D(4), 397-399.
4. Neubauer, W.G., P. Ugincius, and H. Uberall (1969). "Theory of Creeping Waves in Acoustics." *Z. Naturforsch.* 24a, 691-700.
5. Keller, J.B. *Geometrical Theory of Diffraction*. WHOI Lecture Notes.
6. Bird, J.F. (1985). *JOSA A* 2(6):945-953.
7. Born, M. and E. Wolf. *Principles of Optics*, 7th ed., Sec. 8.4.3.
8. Goodman, J. *Introduction to Fourier Optics*, 3rd ed., Ch. 4.
9. Bauer, A.O. (2004). "Some historical and technical aspects of radio navigation, in Germany, over the period 1907 to 1945."
10. Jones, R.V. (1978). *Most Secret War*. Hamish Hamilton.
11. Price, A. (2017). *Instruments of Darkness*. Frontline Books.
