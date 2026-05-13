# Battle of the Beams — ITU Calc Walkthrough

This Jupyter Book walks through the math behind
`ITU_CERTIFIED_Battle_of_the_Beams_Calc2_v9_1_1.xlsx`,
the Aether Cosmology Research Group's ITU-certified Knickebein VHF
propagation calculator.

## Layout

| Chapter | Topic |
|---|---|
| `000_Constants` | Hardcoded physical constants, ground / station / target / path tables, Squint Sandbox defaults. |
| `001_Fock_GE_KL` | ITU-R P.526-16 Fock smooth-Earth diffraction for the Kleve director beam. |
| `002_Fock_GE_ST` | ITU-R P.526-16 Fock smooth-Earth diffraction for the Stollberg cross beam. |
| `003_Sommerfeld_FE_KL` | Sommerfeld-Norton flat-Earth three-term Ez for the Kleve director beam. |
| `004_Sommerfeld_FE_ST` | Sommerfeld-Norton flat-Earth three-term Ez for the Stollberg cross beam. |

All chapters import `common.py`, which is a one-to-one port of the
spreadsheet's ITU sheet (rows 2-3 Fock, rows 6-7 Sommerfeld-Norton),
`const`, `ground`, `stations`, `targets`, `paths_refs` and the
Squint Sandbox defaults (squint 5 deg, W = 99 m, H = 20 m).

## How to use the notebooks

Each chapter is editable. Change the `target` variable in the first
code cell to any name from the `TARGETS` dict (Spalding, Beeston,
Derby, Birmingham, Retford, London, Liverpool, Cardiff, Plymouth,
or the Telefunken TF 400-1000 km sea tests) and re-run the
notebook. Tables and final verdict cells update in place.

The Fock chapters also expose a `Squint Sandbox` cell at the end
where the squint angle, aperture width, and aperture height can be
varied to see how the equisignal corridor width and SNR change.
The defaults reproduce the British-measured 400 to 500 yard
equisignal corridor at Spalding.

## Build the book

```bash
pip install --user jupyter-book
cd /home/alan/claude/BotB/jupyter_book
jupyter-book start    # serves locally
jupyter-book build    # static HTML in _build/html/
```

The notebooks also run standalone in JupyterLab or VS Code; no
Jupyter Book install is required for that.

## Reference

- Spreadsheet: `~/Downloads/ITU_CERTIFIED_Battle_of_the_Beams_Calc2_v9_1_1 1.xlsx`
- Repo: <https://github.com/AlanSpaceAudits/battle-of-the-beams>
- Null hypothesis: `/Null_Hypothesis/Battle_of_the_Beams/Knickebein_Propagation_Null.md`
- GRWAVE methodology: `/Null_Hypothesis/Battle_of_the_Beams/GRWAVE_P368_BotB.md`
