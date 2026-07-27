# Paper-faithful Turner Fig. 3(b)(c) Design

## Goal

Make the independent Fig. 3(b)(c) panels match Turner et al.'s scientific
conventions and visual presentation without replacing independent ED data with
official arrays.

## Scientific convention

The forward-scattering recurrence still traverses the full `L+1` chain from
`Z2` to translated `Z2`. The plotted paper sector is zero momentum and
inversion even, where symmetry-related shells `n` and `L-n` fold together.
Panels (b)(c) therefore contain exactly `L/2+1` shell weights: 11 at L=20 and
17 at L=32. The persisted metadata must distinguish:

- `full_fsa_shell_count = L+1`;
- `plotted_folded_shell_count = L/2+1`;
- `folding = "k=0 inversion-even: n and L-n are symmetry-related"`.

No mirrored `L+1` plotting array may be fabricated.

## State selection

Exact scar states and FSA eigenstates are matched one-to-one by maximizing
their projection in the persisted folded FSA shell subspace, using the existing
paper selector and tolerance.

- Panel (b): the lowest-energy exact member of the matched scar tower.
- Panel (c): the matched nonzero negative-energy tower member with minimum
  absolute energy. Exact zero modes are excluded using the established
  degeneracy tolerance. The negative member fixes the particle-hole tie
  deterministically and matches the paper's “adjacent to E=0” wording.

Each panel records exact/FSA indices, energies, match strength, selection role,
zero-mode tolerance, and shell normalization evidence. Selection must fail
closed if the requested member or one-to-one match is unavailable.

## Rendering

Use paper-like styling for both legacy and independent paths:

- exact shell weights: black circular markers with a solid line;
- FSA shell weights: red cross markers with a dashed line;
- x-axis: folded FSA shell index `n=0..L/2`;
- y-axis: squared shell weight on a linear scale;
- titles: `lowest scar-tower state` and
  `scar-tower state adjacent to E=0`, including the exact energy;
- no DOI substitution. Official data, when available, is a separately labelled
  overlay or mismatch check only.

## Provenance and compatibility

The raw persisted amplitudes and Hamiltonians remain unchanged. The figure
sidecars preserve original arrays, source hashes, stable-snapshot provenance,
generation identity, and transactional publication guarantees. Existing
Fig. 3(a), Fig. 3(d), Fig. 4, server restart, and legacy CLI behavior must not
regress.

## Verification

Tests must establish:

1. official L=32 fixtures expose 17 plotted shells, not 33;
2. local L=20 independent artifacts expose 11 plotted shells and declare the
   21-state full recurrence separately;
3. panel (b) selects the lowest matched tower member;
4. panel (c) excludes zero modes and selects the closest negative matched
   member;
5. black/red styles, labels, axes, and sidecar selection evidence are exact;
6. deliberately permuted or degenerate synthetic spectra cannot silently pick
   a different state;
7. figure publication and server figures-stage validation remain fail-closed.

A real L=20 workflow is rerun and visually inspected. Final paper reproduction
still requires the independent L=32 eigensystem.
