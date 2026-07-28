# L=8 OBC SSH QuSpin ED Smoke Test

## Model

Spinless SSH chain in the `Nf=1` sector with `L=8`, OBC, `t1=0.6`, and `t2=1.0`.

## Local verification

```bash
cd tracks/ed/hubbard-pump
pytest -q test_ssh_ed.py
python run_ssh_ed.py
```

## Cluster pipeline

1. Create `/work/home/mazhuijing/ed-project/tracks/ed/` remotely.
2. Synchronize only this directory with rsync.
3. Verify `python3 -c 'import quspin'` remotely.
4. Validate the exact Slurm request without queueing it.
5. Submit `ssh_ed.slurm` to `xhacnormalb` using one CPU for five minutes.
6. Fetch `results/` without overwriting local data.

## Artifacts

- `results/ssh_spectrum_dos.png`
- `results/energies.csv`
- `results/run_manifest.json`
