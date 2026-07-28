# Notes: Hubbard / Rice-Mele Pump

Free-form domain notes, traps, and known artifacts.

## Sign conventions

- SSH Hamiltonian: H = Σ [t1 c†_A,n c_B,n + t2 c†_B,n c_A,n+1 + h.c.]
- t1 > 0, t2 > 0 by default
- Topological phase: t2 > t1 (winding number ν = 1)
- Trivial phase: t1 > t2 (winding number ν = 0)

## Known finite-size artifacts

- Edge-mode localization length ∝ 1/ln(t2/t1) — for weak dimerization,
  the edge modes spread across many sites and may not be resolved at
  small L.
- At the critical point t1 = t2, the gap closes and finite-size
  rounding becomes severe.

## QuSpin on HPC

- C extension compilation is fragile — see WORKFLOW.md for SCNet
  instructions.
- Python stubs may be needed if C extensions fail to compile.
- Always smoke-test with a small L before submitting batch jobs.

## Resource estimates

- ED memory: D² × 8 bytes, where D = 2^(2L) for spinless fermions
  (Fock space)
- L=8: ~65k × 65k → ~32 MB (laptop-feasible)
- L=10: ~1M × 1M → ~8 GB (cluster recommended)
- L=12: ~16M × 16M → ~2 TB (infeasible without symmetry reduction)
