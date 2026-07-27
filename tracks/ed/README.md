# Exact Diagonalization

## Reproduction target

Turner, Michailidis, Abanin, Serbyn, Papić, "Quantum many-body scars," *Nature Physics* **14**, 745 (2018), [doi:10.1038/s41567-018-0137-5](https://doi.org/10.1038/s41567-018-0137-5), [arXiv:1711.03528](https://arxiv.org/abs/1711.03528).

Reproduce three figures:

1. **Fig. 2** — generic initial states thermalize, but $|Z_2\rangle$ shows anomalous long-time oscillations.
2. **Fig. 3** — rare eigenstates inside the many-body spectrum have anomalously large overlap with $|Z_2\rangle$.
3. **Fig. 4** — level statistics are ETH-like; the revivals are not due to conventional integrability.

## Reproduction scope

- **Fig. 2:** first reproduce the revivals with finite-size QuSpin ED. The
  published figure used thermodynamic-limit iTEBD, so the finite-size result is
  a qualitative reproduction rather than an exact recreation of every curve.
- **Fig. 3:** build and validate the overlap/FSA pipeline at `L=12–20` before
  scaling. The published `L=32`, `k=0`, inversion-even full eigensystem needs a
  high-memory compute node and is not suitable for a 16 GB workstation.
  Panel (d) uses the thesis-semantic FSA selector: match every FSA eigenvector
  one-to-one to the exact eigenstate with maximum
  `|A.T @ U|^2`, energy-sort the complete matched tower, trim one sixth from
  each end, and exclude zero modes from the remaining special set. Projection
  matrices must be finite, and a maximum is rejected as tied when its top-two
  gap is at most `1e-12 + 1e-8*max(|top|,|second|)`. (The smallest DOI gap is
  about `0.223`.) “Other”
  means every nonzero exact state outside the complete untrimmed tower. Both
  plotted `PR2=sum|c|^4` values are unweighted means in the normalized
  `k=0`, inversion-even orbit basis. The DOI archive provides the required
  exact shell amplitudes only at `L=26` and `L=32`; `L=28` and `L=30` are
  explicitly unavailable rather than inferred from Neel overlap. These
  thesis-semantic values do not recover the published orange markers exactly
  (differences reach about 6.6%), so the overlay is not claimed as exact marker
  recovery.
- **Fig. 4:** use the adjacent-gap ratio as an initial diagnostic, then reproduce
  the paper's unfolded level-spacing distribution with its stated spectral
  window. The negative-energy paper window excludes the central zero modes;
  the paper-exact function does not apply a separate zero-mode filter.

Install the locked Python 3.12 environment with:

```bash
uv sync --python 3.12
```

If direct PyPI downloads are slow in mainland China, keep the locked versions
but install their exported requirements through the Tsinghua mirror:

```bash
uv export --frozen --no-hashes --no-emit-project -o /tmp/ed-requirements.txt
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /tmp/ed-requirements.txt
uv sync --offline --frozen --python 3.12
```

The official source data for all figures is published under CC BY 4.0 at
[doi:10.5518/335](https://doi.org/10.5518/335). Local copies belong in the
gitignored directory `.external/official-data/turner-2018/`; they are validation
references, not substitutes for independently generated results.

Download or re-verify all 11 source files with:

```bash
make turner-data
.venv/bin/python scripts/turner2018_data.py --verify-only
```

## Finite-size reproduction pipeline

The tracked scripts implement the small-system validation route before any
high-memory `L=28–32` run:

```bash
# Fig. 2: paper-style Z1/Z2/Z3/Z4 entropy and Z2 ZZ comparison.
# L must be divisible by 12 so every density wave fits the periodic ring.
.venv/bin/python scripts/turner2018_fig2.py --length 12

# Fig. 3: k=0, inversion-even spectrum, Z2 overlap, FSA, and participation ratio
.venv/bin/python scripts/turner2018_fig3.py --length 14

# Fig. 4: k=0, inversion-even gap ratios and unfolded spacing distribution
.venv/bin/python scripts/turner2018_fig4.py --length 16

# Fig. 4 paper-exact: reconstruct L=28/30/32 from official energy spectra,
# compare official xydata, and write NPZ/JSON/PNG validation artifacts
.venv/bin/python scripts/turner2018_fig4.py --paper-exact

# Fig. 4 from recursively validated independent Task 6 spectra
.venv/bin/python scripts/turner2018_fig4.py \
  --independent-results-root tracks/ed/results/turner-2018/task7-independent

# Sparse basis/Hamiltonian resource profile through L=20
.venv/bin/python scripts/turner2018_scaling.py
```

## Thermodynamic-limit Fig. 2 iTEBD

The resumable iTEBD workflow generates the thermodynamic-limit Fig. 2
comparison and numerical diagnostics. The staged targets run `vacuum`, `Z2`,
`Z3`, and `Z4` sequentially, so only one `chi_max=400` infinite MPS is resident
at a time:

```bash
make turner-fig2-itebd-stage1  # evolve all states through t=12
make turner-fig2-itebd-stage2  # resume all states through t=30
```

Each state has a separate HDF5 checkpoint and numerical result below
`tracks/ed/results/turner-2018/fig2-itebd/`. `--resume` restores only a matching
state/configuration checkpoint and does not duplicate the sample at the resume
time. To keep the sample grid unchanged across checkpoint chunks,
`--checkpoint-dt` must be an integer multiple of `--sample-dt`. The paper
figure labels generated iTEBD and official Turner et al. source data separately;
the diagnostics figure records cut-to-cut entropy spread, bond dimension,
discarded weight, and blockade violation. The companion
`fig2_itebd_metrics.json` records the fixed unit-cell entropy cut (`0`), the
linear-fit window (by default `[0, 12]`), configuration fingerprints, official
source hashes, and comparison-grid provenance. Single-state runs use
state-suffixed figure and metrics names. Every state produces diagnostics; the
three-panel paper figure is produced only when `Z2` results are present.

Official source data are required by default. Portable smoke tests may pass
`--allow-missing-official`; those figures receive a visible `INCOMPLETE` banner
and their metrics record the missing files. This flag is not appropriate for a
reported reproduction.

Use a separate output directory for the time-step convergence run:

```bash
.venv/bin/python scripts/turner2018_fig2_itebd.py \
  --state Z2 --target-time 3 --dt 0.025 --chi-max 400 \
  --fit-stop 3 \
  --output-dir tracks/ed/results/turner-2018/fig2-itebd-dt0025
```

Generated arrays and figures are written below
`tracks/ed/results/turner-2018/` and are intentionally gitignored. For small
systems, Fig. 4 uses a lower-half bulk window with the center excluded; it
retains its finite-size unfolding and unit-mean-spacing normalization. The
separate `--paper-exact` route uses the official sorted spectra, cubic
unfolding on `[D/5, D/2-500]`, an exact 50-level trim at each edge, and no
mean-spacing normalization. Full dense diagonalization is intentionally limited to small systems;
the `L=20` check covers constrained-basis construction, sparse Hamiltonian
assembly, one sparse matrix-vector product, and one SciPy `expm_multiply`
time-propagation step.

When the DOI dataset has been restored with `make turner-data`, all three
figure scripts overlay the authors' numerical source data directly from the
official ZIP/HDF5 files. No curves are digitized from the published image.

## Turner L=32 server workflow

The staged server driver uses the native independent engine and supports
`--stage plan|basis|hamiltonian|diagonalize|observables|validate|figures|all`.
Every completed stage is skipped only after its artifact SHA-256 is verified;
manifests also bind the exact predecessor-manifest hashes and scientific plan
hash. A clean stale downstream stage is recomputed by `all`; corrupt artifacts
always fail closed. A direct stage may recompute its own clean-stale output only
when every input is current; stale execution plans or stale/corrupt inputs are
rejected unless an explicit controlled rebuild is requested.
`eigensystem.h5` and
`observables.h5` remain separate, so observables never copy the dense
eigenvector file. Task 6 `all` stops after `validate`; `figures` runs both
independent renderers and publishes its combined stage only after both
transactional generations pass provenance and acceptance checks. Fig. 4
publishes its overview and all size-specific PNG/JSON/NPZ files as one rollback
unit. Production L=28/30/32 additionally requires accepted paper-window
statistics; a local L20 render is explicitly provenance-only because its
91-level exact paper slice is too short for the required 50-level edge trims.
Inspect the resource estimate without computing:

```bash
python3 scripts/turner2018_l32_server.py \
  --length 32 --stage plan --dry-run \
  --output-dir tracks/ed/results/turner-2018/l32-server
```

The production path directly enumerates constrained states and constructs
dihedral orbits; it never scans all `2**L` bitstrings. QuSpin is an optional
small-system cross-check only and is not required by the native engine.

The local SSH alias `qdeshell` targets Dzeshell; connection details remain only
in the user's SSH configuration. The tracked profile records the live
`dzagnormal` scheduler facts and shared project storage. Each immutable wrapper
binds one length class to its exact CPU/GPU/memory request. Validate the
appropriate request only; do not remove `--test-only`:

```bash
# L=22, 24, 26, or 28: 8 CPUs, 1 GPU, 60000M
TURNER_LENGTH=28 scripts/harness_slurm.sh --profile skills/using-slurm/profiles/qdeshell.toml submit --test-only --script scripts/turner2018_dzeshell_l22_28.sbatch

# L=30: 16 CPUs, 2 GPUs, 120000M
TURNER_LENGTH=30 scripts/harness_slurm.sh --profile skills/using-slurm/profiles/qdeshell.toml submit --test-only --script scripts/turner2018_dzeshell_l30.sbatch

# L=32: 32 CPUs, 4 GPUs, 240000M
TURNER_LENGTH=32 scripts/harness_slurm.sh --profile skills/using-slurm/profiles/qdeshell.toml submit --test-only --script scripts/turner2018_dzeshell_l32.sbatch
```

All wrappers request one task for 24 hours and use the exact
`gpu:NVIDIAA80080GBPCIeLC:<count>` GRES. The common runner defaults to the
shared checkout `/work/share/giggleliu/jiangweiqi/quantum.harness`, its
`.venv`, the offline CPython 3.12 runtime below the shared `python/` directory,
and `/work/share/giggleliu/jiangweiqi/results/turner-l<L>`. It rejects a
length/resource mismatch and a virtual environment built from another runtime.
Never embed connection details or credentials in tracked files.

For SCNet, first inspect the live queue configuration; no partition name is
assumed:

```bash
sinfo -o "%P %c %m %G %l %a"
scontrol show partition
```

After selecting values shown by those probes, validate a CPU-only request.
Partition is required; account and QOS are optional and are supplied through
the `sbatch` CLI rather than hard-coded in the provider-neutral script:

```bash
export SCNET_PARTITION="value-from-sinfo"
export SCNET_ACCOUNT=""
export SCNET_QOS=""
export TURNER_LENGTH=32

scnet_args=(--test-only --partition="$SCNET_PARTITION")
[[ -z "$SCNET_ACCOUNT" ]] || scnet_args+=(--account="$SCNET_ACCOUNT")
[[ -z "$SCNET_QOS" ]] || scnet_args+=(--qos="$SCNET_QOS")
sbatch "${scnet_args[@]}" scripts/turner2018_l32_scnet.sbatch
```

The SCNet script requests one node, 64 CPUs, 512 GB, and 24 hours. It does not
request a GPU. If the selected queue requires one, add the exact
probe-confirmed `--gres` value to `scnet_args`; do not guess it. These commands
are probes or test-only validation, not authorization for a real submission.

The Task 6 offline wheelhouse is a local, gitignored staging artifact for
CPython 3.12 on Dzeshell's CentOS 7/glibc 2.17 runtime. Every binary is an
x86-64 `manylinux2014`/`manylinux_2_17` wheel; pure-Python wheels remain
portable. The runtime pins NumPy 2.2.6, SciPy 1.15.3, h5py 3.14.0, and
Matplotlib 3.10.9. The tracked manifest records the exact lock-sourced
filenames, URLs, versions, platform tags, SHA-256 values, lock hash, and
declarative smoke-test schema. Preparation rejects binary wheels with a newer
glibc floor. Prepare or re-verify the binaries with the tracked tool:

```bash
.venv/bin/python scripts/turner2018_wheelhouse.py --prepare --smoke
```

## References

1. **Weiße & Fehske** — classic ED algorithm reference.
   A. Weiße and H. Fehske, "Exact Diagonalization Techniques," in *Computational Many-Particle Physics*, eds. H. Fehske, R. Schneider, A. Weiße, Lecture Notes in Physics 739, pp. 529–544 (Springer, 2008). [doi:10.1007/978-3-540-74686-7_18](https://doi.org/10.1007/978-3-540-74686-7_18).
2. **Sandvik** — best pedagogical spin-ED reference (Ch. 4).
   A. W. Sandvik, "Computational Studies of Quantum Spin Systems," *AIP Conf. Proc.* **1297**, 135–338 (2010). [doi:10.1063/1.3518900](https://doi.org/10.1063/1.3518900), [arXiv:1101.3281](https://arxiv.org/abs/1101.3281).
3. **QuSpin Part I** — Python quick-start for spin chains.
   P. Weinberg and M. Bukov, "QuSpin: a Python Package for Dynamics and Exact Diagonalisation of Quantum Many Body Systems, Part I: Spin Chains," *SciPost Physics* **2**, 003 (2017). [doi:10.21468/SciPostPhys.2.1.003](https://doi.org/10.21468/SciPostPhys.2.1.003), [arXiv:1610.03042](https://arxiv.org/abs/1610.03042), [github](https://github.com/QuSpin/QuSpin).
4. **XDiag** — modern research-grade ED software.
   A. Wietek, L. Staszewski, M. Ulaga, P. L. Ebert, H. Karlsson, S. Sarkar, L. Shackleton, A. Sinha, R. D. Soares, "XDiag: Exact Diagonalization for Quantum Many-Body Systems," *SciPost Physics Codebases* **70** (2026). [doi:10.21468/SciPostPhysCodeb.70](https://doi.org/10.21468/SciPostPhysCodeb.70), [arXiv:2505.02901](https://arxiv.org/abs/2505.02901), [github](https://github.com/awietek/xdiag).
