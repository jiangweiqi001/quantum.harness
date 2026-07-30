# hubbard-pump-2 — Rice-Mele-Hubbard Exact Diagonalization

Exact diagonalization (QuSpin) experiments on the spinful Rice-Mele and Rice-Mele-Hubbard models: topological Chern numbers, Hubbard-U scans, and spin-charge spectrum separation.

Shared library: `rice_mele_ed/` — modular QuSpin model builder, dense diagonalization, I/O utilities.

---

## Quick index

| Experiment | Directory | What it computes |
|---|---|---|
| SSH model ED | `experiments/ssh/` | Spinless SSH chain spectrum, DOS |
| Rice-Mele Chern | `experiments/rice-mele-chern/` | Many-body Chern number via FHS discrete gauge at U=0 |
| Hubbard-U scan | `experiments/u-scan-chern/` | C(U) scan across Hubbard U from −32 to +32 |
| **Spin-charge spectrum** | `experiments/spin-charge/` | Spectral landscape, doublon diagnostic, spin/charge manifold separation |

---

## Spin-charge spectrum separation (primary experiment)

Fixed (U=12, Δ=2), scan dimerisation δ ∈ [−0.5, 0.5].

```bash
cd experiments/spin-charge
# Smoke test (5 δ points)
/opt/anaconda3/bin/python spin_charge_spectrum.py --smoke
# Full scan (41 δ points, L=6 + L=8)
/opt/anaconda3/bin/python spin_charge_spectrum.py --method full_ed
# L=6 only
/opt/anaconda3/bin/python spin_charge_spectrum.py --L 6 --method full_ed
# Run tests
/opt/anaconda3/bin/python -m pytest test_spin_charge_spectrum.py -v
```

**Physics:** Low-energy spin manifold (ΔD_n ≈ 0) separated from high-energy doublon-holon charge band (ΔD_n ~ 1) by ~6.7–7.0 energy units. At δ=0 the spin spectrum is densest; charge band minimum stays finite.

**Key results (41 δ points, full ED):**

| Observable | L=6 (δ=0) | L=8 (δ=0) |
|---|---|---|
| E₀ | −1.552 | −2.032 |
| D₀ (ground-state doublons) | 0.149 | 0.192 |
| Δ_s (spin gap) | 0.254 | 0.186 |
| E_ch^min (charge threshold) | 7.288 | 6.889 |

**Output:** `results/spin-charge/` — CSV/NPZ data + 6 PNG figures.

---

## Hubbard-U scan (Chern number vs U)

```bash
cd experiments/u-scan-chern
/opt/anaconda3/bin/python u_scan_c_solver.py          # full pipeline
/opt/anaconda3/bin/python u_scan_c_solver.py --smoke  # smoke test
/opt/anaconda3/bin/python -m pytest test_u_scan_c_solver.py -v
```

**Output:** `results/u-scan-chern/`

---

## Rice-Mele Chern number (U=0 baseline)

```bash
cd experiments/rice-mele-chern
/opt/anaconda3/bin/python run_rice_mele_chern.py
/opt/anaconda3/bin/python -m pytest test_rice_mele_chern.py -v
```

---

## SSH model ED

```bash
cd experiments/ssh
python run_ssh_ed.py
pytest -q test_ssh_ed.py
```

---

## Shared library: rice_mele_ed

```bash
cd rice_mele_ed
python scripts/run_ed.py --config configs/default.yaml
pytest tests/ -v
```

---

## Cluster submission

All SLURM scripts use partition `xhacnormalb`, account `giggleliu`, remote path `/work/home/mazhuijing/ed-project/`.

| Experiment | SLURM script |
|---|---|
| U-scan Chern | `experiments/u-scan-chern/u_scan_c_solver.slurm` |
| Spin-charge | `experiments/spin-charge/spin_charge_spectrum.slurm` |
| SSH | `experiments/ssh/ssh_ed.slurm` |
| Rice-Mele ED | `rice_mele_ed.slurm` |

---

## Python environment

QuSpin is installed in `/opt/anaconda3/bin/python` (conda base). Always use this Python for experiments:

```bash
/opt/anaconda3/bin/python <script>.py
```
