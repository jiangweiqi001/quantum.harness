# Dzeshell Production Configuration Design

## Goal

Run all medium and large independent PXP ED calculations on the newly
authenticated Dzeshell Slurm service with reproducible, offline, size-scaled
resources and no credentials in Git.

## Security and connection

- Local SSH alias remains `qdeshell`.
- Host, port, username, and private-key path exist only in `~/.ssh/config`.
- The private key remains `~/.ssh/qdeshell_rsa` with mode `0600`.
- No tracked file may contain the hostname, port, username, Windows key path,
  private key, token, or password.
- Tracked cluster profiles contain only the local alias and public scheduler
  facts.

## Live scheduler facts

- Partition: `dzagnormal`.
- Node: 64 CPUs, `515704M` physical memory, eight
  `NVIDIAA80080GBPCIeLC` GPUs.
- CPU/GPU allocation is proportional: 8 CPUs per requested GPU.
- Writable project root: `/work/share/giggleliu/jiangweiqi`.
- Account/QOS observed: account `giggleliu`, QOS `user_jiangweiqi`.
- Login and compute nodes are treated as offline.

## Size-scaled resources

| Length | CPUs | GPUs | Memory | Wall time |
| --- | ---: | ---: | ---: | ---: |
| 22, 24, 26, 28 | 8 | 1 | `60000M` | 24 h |
| 30 | 16 | 2 | `120000M` | 24 h |
| 32 | 32 | 4 | `240000M` | 24 h |

The exact GRES type is `gpu:NVIDIAA80080GBPCIeLC:<count>`. Resource wrappers
are immutable per size class so the submitted request cannot disagree with
`TURNER_LENGTH`. L=28 and L=30 remain acceptance gates before L=32.

## Remote layout and environment

- Checkout: `/work/share/giggleliu/jiangweiqi/quantum.harness`.
- Results: `/work/share/giggleliu/jiangweiqi/results/turner-l<L>`.
- Offline Python: `/work/share/giggleliu/jiangweiqi/python/cpython-3.12`.
- Virtual environment:
  `/work/share/giggleliu/jiangweiqi/quantum.harness/.venv`.
- Wheelhouse and its tracked manifest are verified as an exact file set before
  installation. Runtime fingerprints and artifact manifests remain mandatory.
- The 25 GB HOME filesystem is never used for production eigensystems.

## Submission safety

Configuration may create remote directories, synchronize code/environment, and
run `sbatch --test-only`. A real job is never submitted without displaying the
exact resources and estimated start time and receiving explicit user approval.
Current `--test-only` estimates in 2028 are recorded as a queue blocker; the
service desk should be asked for the hackathon reservation or a usable
CPU/high-memory partition.

## Verification

Tests assert live partition/GRES/resource mappings, absence of secrets, shared
storage defaults, exact offline environment verification, and rejection of
unsupported lengths. Remote checks verify SSH, Slurm tools, storage
writability, Python 3.12 package versions, code fingerprint, and one
non-submitting `--test-only` request per size class.
