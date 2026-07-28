# Prior Runs: Hubbard / Rice-Mele Pump

Results already produced by this harness. Do not recompute these
without a reason — use them as anchors.

## Run: 2026-07-28-ssh-smoke

- **Model:** SSH, spinless fermion chain
- **Parameters:** t1=1, t2=1.5, L=8 (4 unit cells), OBC
- **Method:** QuSpin exact diagonalization, Python
- **Tool:** quspin v1.0.1
- **Key results:**
  - Two zero-energy edge modes detected (E ≈ 0 within machine precision)
  - Bulk gap consistent with 2|t1-t2| = 1.0
- **Status:** Verified — matches analytic SSH solution
- **Location:** `results/ssh-smoke/`
