# LASG02 Production Configuration Design

## Goal

Use the `student090` CPU account for independent Turner ED at L=22–30 and for
CPU iTEBD work, while retaining the existing Dzeshell account for the L=32
high-memory solve.

## Security and storage

- The local SSH alias is `lasg02-student090`.
- Host, port, username, and private-key path remain only in `~/.ssh/config`.
- No credential or private connection detail is tracked by Git.
- The reviewed checkout is `/public/home/student090/quantum.harness`.
- Results are written below `/public/home/student090/results`.
- The portable CPython 3.12 runtime and exact offline wheelhouses live below
  `/public/home/student090/python` and `/public/home/student090/wheelhouse`.

## Scheduler facts

- CPU partition: `ihicnormal`.
- Account: `chenkun2025`; QOS: `user_student090`.
- Standard nodes expose 28 CPUs and approximately 94 GB RAM.
- `DefMemPerCPU` is 3363 MB.
- L=22, 24, 26, 28, and 30 use one task, 24 CPUs, `80000M`, and 24 hours.
- L=32 is rejected because the dense eigensolver requires approximately
  240 GB; it remains assigned to Dzeshell.

## Execution and publication

One immutable LASG02 wrapper accepts only L=22–30 even lengths and invokes a
shared runner from the canonical checkout. The runner validates the portable
runtime, exact package versions, Slurm resources, and approved paths before
executing all restartable ED stages. Each length has an isolated result
directory and atomic stage manifests. L=28 and L=30 additionally feed Fig. 4;
all five sizes feed Fig. 3(d).

The same account may run four independent iTEBD continuation jobs after a
separate TeNPy-compatible offline runtime and state-specific checkpoint
contract pass their tests. ED and iTEBD never write the same output directory.

## Submission safety

Run `sbatch --test-only` first, then submit real jobs because the user has
explicitly authorized this execution sequence. Record every job ID, requested
resource, state, reason, and output path. A distant scheduler estimate is not
reported as execution progress; queued jobs remain pending until Slurm
allocates resources. Failed environment checks must terminate before numerical
work begins.

