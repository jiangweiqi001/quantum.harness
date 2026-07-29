#!/usr/bin/env python3
"""Coarse scan and adaptive refinement for charge gapless point(s) in the
Rice-Mele-Hubbard model at small U.

Usage:
    python scripts/run_scan.py --config configs/coarse_L6.yaml --mode auto
    python scripts/run_scan.py --config configs/coarse_L6.yaml --mode coarse
    python scripts/run_scan.py --config configs/coarse_L6.yaml --mode refine --U 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_RMH_GAP = _PROJECT.parent / "rmh_gap_landscape"
sys.path.insert(0, str(_RMH_GAP))

from src.gaps import GapPointResult, solve_point  # noqa: E402

RESULTS_ROOT = _PROJECT.parent.parent / "results" / "gapless-point-split"


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def build_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    delta = np.linspace(cfg["delta_min"], cfg["delta_max"], cfg["n_delta"])
    Delta = np.linspace(cfg["Delta_min"], cfg["Delta_max"], cfg["n_Delta"])
    return delta, Delta


def build_refine_grid(center_delta: float, center_Delta: float,
                      cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    ref = cfg["refine"]
    dw = ref["delta_window"]
    Dw = ref["Delta_window"]
    delta = np.linspace(center_delta - dw, center_delta + dw, ref["n_delta_fine"])
    Delta = np.linspace(center_Delta - Dw, center_Delta + Dw, ref["n_Delta_fine"])
    return delta, Delta


# ---------------------------------------------------------------------------
# Minima detection (used by both scan and analyze)
# ---------------------------------------------------------------------------

def detect_local_minima(delta_grid: np.ndarray, Delta_grid: np.ndarray,
                        gap_map: np.ndarray, eps: float = 0.01,
                        merge_distance: float = 2.0) -> list[dict]:
    """Detect local minima in a 2D gap map with sub-pixel refinement.

    Parameters
    ----------
    delta_grid, Delta_grid : 1D arrays
    gap_map : 2D array, shape (n_delta, n_Delta)
    eps : float
        Only return minima with gap < eps.
    merge_distance : float
        Merge minima closer than this many grid spacings.

    Returns
    -------
    list of dicts with keys: delta, Delta, gap, Hessian_eigvals
    """
    from scipy.ndimage import gaussian_filter

    n_d, n_D = gap_map.shape
    if n_d < 3 or n_D < 3:
        return []

    d_delta = delta_grid[1] - delta_grid[0]
    d_Delta = Delta_grid[1] - Delta_grid[0]

    # Light smoothing
    smoothed = gaussian_filter(gap_map, sigma=1.0, mode='nearest')

    # Find interior points strictly less than all 8 neighbours
    candidates = []
    for i in range(1, n_d - 1):
        for j in range(1, n_D - 1):
            val = smoothed[i, j]
            patch = smoothed[i-1:i+2, j-1:j+2]
            if val < np.min(np.delete(patch.ravel(), 4)):  # 4 = centre index
                # Sub-pixel refinement via quadratic fit on raw gap_map
                refined = _quadratic_refine_2d(
                    delta_grid, Delta_grid, gap_map, i, j)
                if refined is not None and refined["gap"] < eps:
                    candidates.append(refined)

    # Cluster nearby minima
    merged = _cluster_minima(candidates, d_delta, d_Delta, merge_distance)
    return merged


def _quadratic_refine_2d(delta_grid, Delta_grid, gap_map, i, j):
    """Fit a*x² + b*y² + c*x*y + d*x + e*y + f to 3x3 patch, return minimum."""
    n_d, n_D = gap_map.shape
    if i < 1 or i >= n_d - 1 or j < 1 or j >= n_D - 1:
        return None

    xs = delta_grid[i-1:i+2]
    ys = Delta_grid[j-1:j+2]
    zs = gap_map[i-1:i+2, j-1:j+2]

    # Build design matrix for 6-parameter fit on 9 points
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    A = np.column_stack([X.ravel()**2, Y.ravel()**2, (X * Y).ravel(),
                          X.ravel(), Y.ravel(), np.ones(9)])
    b = zs.ravel()

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    a, b2, c, d, e, f = coeffs  # noqa: E741

    # Solve for stationary point: 2a*x + c*y + d = 0, 2b2*y + c*x + e = 0
    M = np.array([[2 * a, c], [c, 2 * b2]])
    rhs = np.array([-d, -e])

    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # Fall back to grid point
        sol = np.array([delta_grid[i], Delta_grid[j]])

    delta_min = float(sol[0])
    Delta_min = float(sol[1])

    # Evaluate gap at minimum
    gap_min = float(a * delta_min**2 + b2 * Delta_min**2 +
                    c * delta_min * Delta_min + d * delta_min + e * Delta_min + f)

    # Hessian eigenvalues for curvature
    H = np.array([[2 * a, c], [c, 2 * b2]])
    eigvals = np.linalg.eigvalsh(H)

    return {
        "delta": delta_min,
        "Delta": Delta_min,
        "gap": gap_min,
        "hessian_eigvals": eigvals.tolist(),
    }


def _cluster_minima(candidates: list[dict], d_delta: float, d_Delta: float,
                    merge_distance: float) -> list[dict]:
    """Merge minima closer than merge_distance grid spacings."""
    if len(candidates) <= 1:
        return candidates

    # Simple agglomerative clustering
    n = len(candidates)
    merged = [True] * n
    result = []

    for i in range(n):
        if not merged[i]:
            continue
        group = [candidates[i]]
        for j in range(i + 1, n):
            if not merged[j]:
                continue
            dx = (candidates[j]["delta"] - candidates[i]["delta"]) / d_delta
            dy = (candidates[j]["Delta"] - candidates[i]["Delta"]) / d_Delta
            if np.sqrt(dx**2 + dy**2) < merge_distance:
                group.append(candidates[j])
                merged[j] = False

        # Keep the one with smallest gap
        best = min(group, key=lambda m: m["gap"])
        result.append(best)

    # Sort by gap value
    result.sort(key=lambda m: m["gap"])
    return result


# ---------------------------------------------------------------------------
# Single-U scan
# ---------------------------------------------------------------------------

def scan_coarse_grid(L: int, U: float, delta_grid: np.ndarray,
                     Delta_grid: np.ndarray, out_dir: Path,
                     method: str = "auto") -> Path:
    """Scan the full coarse grid for one (L, U) and save results.

    Returns path to the saved NPZ file.
    """
    n_d = len(delta_grid)
    n_D = len(Delta_grid)
    n_total = n_d * n_D
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allocate arrays
    Delta_c_map = np.full((n_d, n_D), np.nan)
    Delta_s_map = np.full((n_d, n_D), np.nan)
    Delta_MB_map = np.full((n_d, n_D), np.nan)
    E0_map = np.full((n_d, n_D), np.nan)

    print(f"  Grid: {n_d}×{n_D} = {n_total} points  method={method}")
    t_start = time.perf_counter()
    n_ok = 0
    n_fail = 0

    for idx_d in range(n_d):
        delta = float(delta_grid[idx_d])
        for idx_D in range(n_D):
            Dv = float(Delta_grid[idx_D])
            flat_idx = idx_d * n_D + idx_D

            try:
                r = solve_point(L=L, delta=delta, Delta=Dv, U=U, method=method)
                Delta_c_map[idx_d, idx_D] = r.Delta_c
                Delta_s_map[idx_d, idx_D] = r.Delta_s
                Delta_MB_map[idx_d, idx_D] = r.Delta_MB
                E0_map[idx_d, idx_D] = r.E0_half
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                if n_fail <= 5:
                    print(f"    FAIL δ={delta:+.4f} Δ={Dv:+.4f}: {exc}")

            if (flat_idx + 1) % max(1, n_total // 20) == 0:
                pct = (flat_idx + 1) / n_total * 100
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (flat_idx + 1) * (n_total - flat_idx - 1)
                print(f"    {pct:.0f}% ({flat_idx+1}/{n_total})  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed = time.perf_counter() - t_start
    valid_mask = ~np.isnan(Delta_c_map)
    print(f"  Done: {n_ok} OK, {n_fail} failed, {elapsed:.0f}s")
    if n_ok > 0:
        print(f"  Δc: min={np.min(Delta_c_map[valid_mask]):.6f}  "
              f"max={np.max(Delta_c_map[valid_mask]):.4f}")

    # Save
    tag = f"U{U:.3f}".replace(".", "p")
    out_path = out_dir / f"gaps_{tag}.npz"
    np.savez_compressed(
        out_path,
        delta_values=delta_grid,
        Delta_values=Delta_grid,
        Delta_c=Delta_c_map,
        Delta_s=Delta_s_map,
        Delta_MB=Delta_MB_map,
        E0_half=E0_map,
        L=np.array([L]),
        U=np.array([U]),
        wall_time_s=np.array([elapsed]),
        n_ok=np.array([n_ok]),
        n_fail=np.array([n_fail]),
    )
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------

def refine_minima(L: int, U: float, coarse_path: Path, cfg: dict,
                  out_dir: Path) -> list[dict]:
    """Refine around detected minima in a coarse scan result.

    Returns list of refined minimum dicts.
    """
    ref_cfg = cfg["refine"]
    method = cfg.get("method", "auto")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load coarse data
    data = np.load(coarse_path, allow_pickle=False)
    delta_grid = data["delta_values"]
    Delta_grid = data["Delta_values"]
    Delta_c = data["Delta_c"]

    # Detect minima
    minima = detect_local_minima(
        delta_grid, Delta_grid, Delta_c,
        eps=ref_cfg["eps"],
        merge_distance=ref_cfg["merge_distance"],
    )
    print(f"  Detected {len(minima)} candidate minima:")
    for k, m in enumerate(minima):
        print(f"    min {k}: δ={m['delta']:+.6f}  Δ={m['Delta']:+.6f}  "
              f"Δc={m['gap']:.6f}  Hess eVals={m['hessian_eigvals']}")

    if not minima:
        print("  No minima found — skipping refinement.")
        return []

    refined = []
    for k, m in enumerate(minima):
        print(f"\n  Refining minimum {k} at δ={m['delta']:+.6f} Δ={m['Delta']:+.6f} ...")
        d_fine, D_fine = build_refine_grid(m["delta"], m["Delta"], cfg)
        n_d = len(d_fine)
        n_D = len(D_fine)

        Delta_c_fine = np.full((n_d, n_D), np.nan)
        t0 = time.perf_counter()
        n_ok = 0

        for i, delta in enumerate(d_fine):
            for j, Dv in enumerate(D_fine):
                try:
                    r = solve_point(L=L, delta=float(delta), Delta=float(Dv),
                                    U=U, method=method)
                    Delta_c_fine[i, j] = r.Delta_c
                    n_ok += 1
                except Exception:
                    pass

        elapsed = time.perf_counter() - t0

        # Find refined minimum
        fine_minima = detect_local_minima(
            d_fine, D_fine, Delta_c_fine,
            eps=ref_cfg["eps"] * 2,  # looser threshold for refinement
            merge_distance=ref_cfg["merge_distance"],
        )

        # Save refinement result
        tag = f"U{U:.3f}_min{k}".replace(".", "p")
        out_path = out_dir / f"refine_{tag}.npz"
        np.savez_compressed(
            out_path,
            delta_values=d_fine,
            Delta_values=D_fine,
            Delta_c=Delta_c_fine,
            L=np.array([L]),
            U=np.array([U]),
            center_delta=np.array([m["delta"]]),
            center_Delta=np.array([m["Delta"]]),
            wall_time_s=np.array([elapsed]),
        )
        print(f"    Saved: {out_path}  ({n_ok} OK, {elapsed:.0f}s)")

        if fine_minima:
            best = fine_minima[0]
            print(f"    Refined min: δ={best['delta']:+.6f}  Δ={best['Delta']:+.6f}  "
                  f"Δc={best['gap']:.6f}")
            refined.append(best)
        else:
            print(f"    No minimum found in refined grid — keeping coarse estimate")
            refined.append(m)

    return refined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Charge gapless point scan — RMH model at small U")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--mode", default="auto",
                        choices=["coarse", "refine", "auto"],
                        help="Scan mode (default: auto = coarse + refine)")
    parser.add_argument("--U", type=float, default=None,
                        help="Single U value (overrides config U_list)")
    parser.add_argument("--U-only", type=float, nargs="*", default=None,
                        help="Specific U values to run (overrides config)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    L = cfg["L"]
    method = cfg.get("method", "auto")

    # Determine U list
    if args.U_only:
        U_list = args.U_only
    elif args.U is not None:
        U_list = [args.U]
    else:
        U_list = cfg["U_list"]

    delta_grid, Delta_grid = build_grid(cfg)
    base_out = RESULTS_ROOT / f"L{L}"

    print("=" * 64)
    print(f"GAPLESS POINT SPLIT SCAN")
    print(f"  L = {L}")
    print(f"  U_list = {U_list}")
    print(f"  δ ∈ [{delta_grid[0]:.2f}, {delta_grid[-1]:.2f}]  "
          f"n={len(delta_grid)}")
    print(f"  Δ ∈ [{Delta_grid[0]:.2f}, {Delta_grid[-1]:.2f}]  "
          f"n={len(Delta_grid)}")
    print(f"  mode = {args.mode}  method = {method}")
    print(f"  results → {base_out}")
    print("=" * 64)

    t_total = time.perf_counter()

    for U in U_list:
        print(f"\n{'─' * 48}")
        print(f"U = {U:.3f}")
        print(f"{'─' * 48}")

        # --- Coarse scan ---
        if args.mode in ("coarse", "auto"):
            coarse_dir = base_out / "coarse"
            coarse_path = scan_coarse_grid(
                L, U, delta_grid, Delta_grid, coarse_dir, method=method)

        # --- Refine ---
        if args.mode in ("refine", "auto"):
            tag = f"U{U:.3f}".replace(".", "p")
            coarse_path = base_out / "coarse" / f"gaps_{tag}.npz"
            if not coarse_path.exists():
                print(f"  ERROR: coarse scan not found at {coarse_path}")
                print(f"  Run with --mode coarse first.")
                continue

            refine_dir = base_out / "refine"
            refined = refine_minima(L, U, coarse_path, cfg, refine_dir)

            # Save refined minima summary
            if refined:
                tag = f"U{U:.3f}".replace(".", "p")
                summary_path = base_out / "refine" / f"minima_{tag}.csv"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, "w") as fh:
                    fh.write("index,delta,Delta,gap,hessian_eigval_0,"
                             "hessian_eigval_1\n")
                    for k, m in enumerate(refined):
                        fh.write(f"{k},{m['delta']:.8f},{m['Delta']:.8f},"
                                 f"{m['gap']:.8f},"
                                 f"{m['hessian_eigvals'][0]:.6f},"
                                 f"{m['hessian_eigvals'][1]:.6f}\n")
                print(f"  Minima summary: {summary_path}")

    elapsed = time.perf_counter() - t_total
    print(f"\n{'=' * 64}")
    print(f"All scans complete. Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results: {base_out}")
    print("Done.")


if __name__ == "__main__":
    main()
