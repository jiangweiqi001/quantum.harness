import ast
import inspect
import tracemalloc

import numpy as np
import pytest
import scipy.sparse as sp

import turner2018_ed_observables as observables_module
import turner2018_ed_validation as validation_module
from pxp_ed import (
    constrained_basis,
    density_wave_state,
    pxp_hamiltonian,
    symmetry_basis_k0_inversion_even,
)
from turner2018_ed_engine import assemble_reduced_hamiltonian, build_orbit_basis
from turner2018_ed_observables import (
    compare_degenerate_invariants,
    compute_observables,
    project_product_state,
    stream_fsa_shells,
    stream_pr2,
)
from turner2018_fig3 import fsa_basis


def _align_global_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    overlap = float(np.dot(reference, candidate))
    if overlap < 0.0:
        return -candidate
    return candidate


def _apply_shell_sign_similarity(
    hamiltonian: np.ndarray,
    shell_signs: np.ndarray,
) -> np.ndarray:
    return shell_signs[:, None] * hamiltonian * shell_signs[None, :]


def test_shell_sign_similarity_transform_changes_signed_offdiagonals():
    hamiltonian = np.asarray([[0.0, 2.0], [2.0, 0.0]])
    signs = np.asarray([1.0, -1.0])
    np.testing.assert_array_equal(
        _apply_shell_sign_similarity(hamiltonian, signs),
        np.asarray([[0.0, -2.0], [-2.0, 0.0]]),
    )


def _oracle_project_shells(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orbit = build_orbit_basis(length)
    reduced = assemble_reduced_hamiltonian(orbit).toarray()
    constrained = constrained_basis(length, pbc=True)
    hamiltonian = pxp_hamiltonian(constrained, length, pbc=True)
    z2 = density_wave_state(length, 2)
    shells_full, beta = fsa_basis(hamiltonian, constrained, z2, length)

    transform = symmetry_basis_k0_inversion_even(constrained, length).toarray()
    projected_complex = (transform.T @ shells_full.T).T[: length // 2 + 1]
    np.testing.assert_allclose(projected_complex.imag, 0.0, atol=1e-14, rtol=0.0)
    projected_shells = np.asarray(projected_complex.real, dtype=np.float64)
    projected_shells /= np.linalg.norm(projected_shells, axis=1, keepdims=True)
    projected_h = projected_shells @ reduced @ projected_shells.T
    projected_h = 0.5 * (projected_h + projected_h.T)
    return np.asarray(beta, dtype=np.float64), projected_shells, projected_h


def test_z2_sector_projection_has_half_weight():
    basis = build_orbit_basis(12)
    z2 = project_product_state(basis, density_wave_state(12, 2))
    assert np.vdot(z2, z2).real == pytest.approx(0.5)


def test_streamed_pr2_matches_definition():
    vectors = np.linalg.qr(np.arange(1, 26, dtype=float).reshape(5, 5))[0]
    np.testing.assert_allclose(
        stream_pr2(vectors, chunk_columns=2),
        np.sum(np.abs(vectors) ** 4, axis=0),
    )


def test_streamed_pr2_preserves_complex_coefficients():
    vectors = np.asarray(
        [[1.0j / np.sqrt(2.0), 0.5 + 0.5j], [1.0 / np.sqrt(2.0), 0.5 - 0.5j]]
    )
    np.testing.assert_allclose(
        stream_pr2(vectors, chunk_columns=1),
        np.sum(np.abs(vectors) ** 4, axis=0),
    )


@pytest.mark.parametrize("chunk_columns", [0, -1, 1.5, True, np.bool_(False)])
def test_streamed_pr2_rejects_invalid_chunk_columns(chunk_columns):
    with pytest.raises((TypeError, ValueError), match="chunk_columns"):
        stream_pr2(np.eye(2), chunk_columns=chunk_columns)


@pytest.mark.parametrize("length", [10, 12, 14, 16])
def test_streamed_fsa_matches_full_basis_oracle(length: int):
    orbit = build_orbit_basis(length)
    z2 = density_wave_state(length, 2)
    streamed = stream_fsa_shells(orbit, z2)
    oracle_beta, oracle_shells, oracle_h = _oracle_project_shells(length)

    np.testing.assert_allclose(streamed.beta, oracle_beta, atol=1e-12, rtol=0.0)

    shell_signs = np.ones(oracle_shells.shape[0], dtype=np.float64)
    for shell_index in range(oracle_shells.shape[0]):
        aligned = _align_global_sign(
            oracle_shells[shell_index], streamed.projected_shells[shell_index]
        )
        if np.dot(oracle_shells[shell_index], streamed.projected_shells[shell_index]) < 0:
            shell_signs[shell_index] = -1.0
        np.testing.assert_allclose(
            aligned,
            oracle_shells[shell_index],
            atol=1e-12,
            rtol=0.0,
        )

    np.testing.assert_allclose(
        streamed.reduced_fsa_hamiltonian,
        _apply_shell_sign_similarity(oracle_h, shell_signs),
        atol=1e-11,
        rtol=0.0,
    )


def test_streamed_fsa_is_invariant_to_small_chunks():
    orbit = build_orbit_basis(12)
    z2 = density_wave_state(12, 2)
    baseline = stream_fsa_shells(orbit, z2, chunk_size=4096)

    for chunk_size in (1, 3, 17):
        candidate = stream_fsa_shells(orbit, z2, chunk_size=chunk_size)
        np.testing.assert_allclose(candidate.beta, baseline.beta, atol=1e-13, rtol=0.0)
        np.testing.assert_allclose(
            candidate.projected_shells,
            baseline.projected_shells,
            atol=1e-13,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            candidate.reduced_fsa_hamiltonian,
            baseline.reduced_fsa_hamiltonian,
            atol=1e-13,
            rtol=0.0,
        )
        assert candidate.max_source_chunk <= chunk_size
        assert candidate.max_projection_chunk <= chunk_size
        assert candidate.distance_dtype == "uint8"


def test_l20_streamed_recurrence_and_folded_plot_shell_counts_are_distinct():
    length = 20
    orbit = build_orbit_basis(length)
    streamed = stream_fsa_shells(
        orbit,
        density_wave_state(length, 2),
        chunk_size=1024,
    )

    assert len(streamed.beta) + 1 == length + 1
    assert streamed.projected_shells.shape[0] == length // 2 + 1 == 11
    assert streamed.reduced_fsa_hamiltonian.shape == (11, 11)


def test_streamed_fsa_has_no_scalar_per_state_propagation_loop():
    tree = ast.parse(inspect.getsource(stream_fsa_shells))
    forbidden_state_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.For, ast.AsyncFor))
        and any(
            isinstance(name, ast.Name) and name.id == "support"
            for name in ast.walk(node.iter)
        )
    ]
    assert not forbidden_state_loops, "FSA propagation must not loop over support"

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "searchsorted"
        ):
            continue
        ancestor = parents.get(node)
        inside_loop = False
        while ancestor is not None:
            inside_loop = inside_loop or isinstance(ancestor, (ast.For, ast.AsyncFor))
            ancestor = parents.get(ancestor)
        if inside_loop:
            assert isinstance(node.args[1], ast.Name)
            assert node.args[1].id == "destination_states"


def test_projection_canonicalizes_full_basis_in_bounded_chunks(monkeypatch):
    orbit = build_orbit_basis(10)
    observed_sizes = []
    original = observables_module.canonical_dihedral

    def recording_canonical(states, length):
        observed_sizes.append(len(states))
        return original(states, length)

    monkeypatch.setattr(observables_module, "canonical_dihedral", recording_canonical)
    stream_fsa_shells(
        orbit,
        density_wave_state(10, 2),
        max_shell=0,
        chunk_size=7,
    )

    assert max(observed_sizes) <= 7
    assert sum(observed_sizes) == len(orbit.constrained_states)


@pytest.mark.parametrize("max_shell", [-1, 13, 1.5, True, np.int64(-2)])
def test_streamed_fsa_rejects_invalid_max_shell(max_shell):
    orbit = build_orbit_basis(12)
    with pytest.raises((TypeError, ValueError), match="max_shell"):
        stream_fsa_shells(
            orbit,
            density_wave_state(12, 2),
            max_shell=max_shell,
            chunk_size=7,
        )


def test_degenerate_invariant_comparison_uses_projectors_not_vector_pr2():
    energies = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    reference = np.eye(3, dtype=np.float64)
    angle = np.pi / 4.0
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    candidate = reference @ rotation
    z2 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

    diagnostics = compare_degenerate_invariants(
        reference_energies=energies,
        reference_vectors=reference,
        candidate_energies=energies,
        candidate_vectors=candidate,
        reference_z2_sector_state=z2,
        candidate_z2_sector_state=z2,
        chunk_columns=2,
    )

    assert diagnostics["max_total_z2_diff"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["max_projector_diag_diff"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["max_subspace_sine"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["degenerate_group_count"] == 1
    assert diagnostics["isolated_group_count"] == 1
    # Rotating inside the degenerate two-state manifold changes vector-level PR2.
    assert diagnostics["max_pr2_diff_isolated"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["validation_chunk_columns"] == 2
    assert diagnostics["orthogonality_sample_count"] > 0


def test_degenerate_rotated_subspace_residual_is_stable():
    rng = np.random.default_rng(417)
    reference = np.linalg.qr(rng.normal(size=(48, 16)))[0]
    rotation = np.linalg.qr(rng.normal(size=(16, 16)))[0]
    candidate = reference @ rotation
    energies = np.zeros(16)
    z2 = reference[:, 0]

    diagnostics = compare_degenerate_invariants(
        reference_energies=energies,
        reference_vectors=reference,
        candidate_energies=energies,
        candidate_vectors=candidate,
        reference_z2_sector_state=z2,
        candidate_z2_sector_state=z2,
        chunk_columns=3,
        orthogonality_samples=31,
    )

    assert diagnostics["max_subspace_sine"] <= 1e-12
    assert diagnostics["subspace_chunk_columns"] == 3


def test_default_l32_pair_sampling_spans_early_middle_and_late_spectrum():
    dimension = 77_436
    batches = validation_module.sample_pair_batches(
        dimension,
        sample_count=4096,
        batch_size=256,
    )
    first_indices = np.concatenate([left for left, _right in batches])

    assert np.any(first_indices < dimension // 3)
    assert np.any(
        (first_indices >= dimension // 3)
        & (first_indices < 2 * dimension // 3)
    )
    assert np.any(first_indices >= 2 * dimension // 3)


def test_default_sampling_detects_nonorthogonal_late_spectrum_pair():
    dimension = 192
    late_pair = next(
        (int(first), int(second))
        for left, right in validation_module.sample_pair_batches(
            dimension,
            sample_count=4096,
            batch_size=17,
        )
        for first, second in zip(left, right)
        if first >= 2 * dimension // 3
    )
    vectors = np.eye(dimension)
    vectors[:, late_pair[1]] = vectors[:, late_pair[0]]

    with pytest.raises(ValueError, match="orthonormal"):
        validation_module.validate_orthonormal_columns(
            vectors,
            chunk_columns=17,
            orthogonality_samples=4096,
            tolerance=1e-12,
            name="late-spectrum",
        )


def test_pair_sampling_stays_lazy_and_batch_bounded_for_huge_sample_count():
    dimension = 1_000_000
    total_pairs = dimension * (dimension - 1) // 2

    tracemalloc.start()
    batches = validation_module.sample_pair_batches(
        dimension,
        sample_count=total_pairs,
        batch_size=23,
    )
    left, right = next(batches)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert left.shape == right.shape == (23,)
    assert np.all(left != right)
    assert peak < 1_000_000


def test_degenerate_invariants_detect_same_diagonal_different_subspace():
    energies = np.asarray([0.0])
    reference = np.asarray([[1.0], [1.0]]) / np.sqrt(2.0)
    candidate = np.asarray([[1.0], [-1.0]]) / np.sqrt(2.0)
    z2 = np.asarray([1.0, 0.0])

    diagnostics = compare_degenerate_invariants(
        reference_energies=energies,
        reference_vectors=reference,
        candidate_energies=energies,
        candidate_vectors=candidate,
        reference_z2_sector_state=z2,
        candidate_z2_sector_state=z2,
    )

    assert diagnostics["max_projector_diag_diff"] == pytest.approx(0.0, abs=1e-14)
    assert diagnostics["max_subspace_sine"] == pytest.approx(1.0, abs=1e-14)


def test_validation_paths_have_no_full_square_gram_or_identity():
    source = "\n".join(
        (
            inspect.getsource(observables_module),
            inspect.getsource(validation_module),
        )
    )
    tree = ast.parse(source)

    forbidden_eye_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in {"eye", "identity"}
    ]
    assert not forbidden_eye_calls

    forbidden_self_grams = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.MatMult):
            continue
        left = ast.unparse(node.left).replace(".conj()", "")
        right = ast.unparse(node.right)
        if left.endswith(".T") and left[:-2] == right:
            forbidden_self_grams.append(ast.unparse(node))
    assert not forbidden_self_grams


def test_degenerate_invariants_reject_candidate_partition_mismatch():
    reference_energies = np.asarray([0.0, 0.5e-10, 1.0])
    candidate_energies = np.asarray([0.0, 1.5e-10, 1.0])
    vectors = np.eye(3)
    z2 = np.asarray([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="degeneracy partitions"):
        compare_degenerate_invariants(
            reference_energies=reference_energies,
            reference_vectors=vectors,
            candidate_energies=candidate_energies,
            candidate_vectors=vectors,
            reference_z2_sector_state=z2,
            candidate_z2_sector_state=z2,
            tolerance=1e-10,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("reference_energies", np.asarray([[0.0, 1.0]]), "one-dimensional"),
        ("reference_energies", np.asarray([1.0, 0.0]), "sorted"),
        ("candidate_energies", np.asarray([0.0, np.inf]), "finite"),
        ("reference_vectors", np.ones(2), "two-dimensional"),
        ("candidate_vectors", np.asarray([[1.0, 0.0], [0.0, np.nan]]), "finite"),
        ("reference_vectors", np.asarray([[1.0, 1.0], [0.0, 0.0]]), "orthonormal"),
        ("candidate_z2_sector_state", np.asarray([0.0, 1.0]), "Z2"),
        ("tolerance", 0.0, "tolerance"),
        ("tolerance", True, "tolerance"),
        ("chunk_columns", 0, "chunk_columns"),
        ("chunk_columns", True, "chunk_columns"),
        ("orthogonality_samples", 0, "orthogonality_samples"),
        ("orthogonality_samples", True, "orthogonality_samples"),
    ],
)
def test_degenerate_invariants_reject_invalid_inputs(field, replacement, message):
    arguments = {
        "reference_energies": np.asarray([0.0, 1.0]),
        "reference_vectors": np.eye(2),
        "candidate_energies": np.asarray([0.0, 1.0]),
        "candidate_vectors": np.eye(2),
        "reference_z2_sector_state": np.asarray([1.0, 0.0]),
        "candidate_z2_sector_state": np.asarray([1.0, 0.0]),
        "tolerance": 1e-10,
    }
    arguments[field] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        compare_degenerate_invariants(**arguments)


def test_compute_observables_consumes_eigensystem_without_dense_solve(monkeypatch):
    basis = build_orbit_basis(10)
    hamiltonian = assemble_reduced_hamiltonian(basis)
    energies, eigenvectors = np.linalg.eigh(hamiltonian.toarray())

    def forbidden(*args, **kwargs):
        raise AssertionError("compute_observables must not densify or diagonalize")

    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    result = compute_observables(
        basis,
        hamiltonian,
        energies,
        eigenvectors,
        chunk_size=3,
        chunk_columns=3,
    )

    np.testing.assert_array_equal(result["energies"], energies)
    np.testing.assert_array_equal(result["eigenvectors"], eigenvectors)
    np.testing.assert_allclose(np.sum(result["overlap_z2"]), 0.5, atol=1e-12)
    assert "invariant_diagnostics" not in result
    assert result["fsa_shell_vectors_sector"].shape == (6, len(basis.representatives))
    assert result["validation_metadata"]["chunk_columns"] == 3
    assert result["validation_metadata"]["residual_columns_checked"] == len(energies)
    assert result["validation_metadata"]["orthogonality_sample_count"] > 0
    assert result["validation_metadata"]["orthogonality_sampling_policy"] == (
        "midpoint-stratified first indices with SplitMix64 partner slots when "
        "samples <= D-1; otherwise midpoint-stratified canonical pair ranks"
    )
    assert result["validation_metadata"]["orthogonality_pair_batch_size"] == 3
    assert result["validation_metadata"]["orthogonality_pair_metadata_memory"] == (
        "bounded by two int64 arrays of pair_batch_size"
    )


def test_compute_observables_rejects_nonhermitian_matrix():
    basis = build_orbit_basis(10)
    hamiltonian = assemble_reduced_hamiltonian(basis)
    energies, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
    broken = hamiltonian.copy().tolil()
    broken[0, 1] += 0.25

    with pytest.raises(ValueError, match="Hermitian"):
        compute_observables(
            basis,
            broken.tocsr(),
            energies,
            eigenvectors,
            chunk_size=4,
        )


def test_compute_observables_rejects_orthonormal_wrong_eigenpairs():
    basis = build_orbit_basis(10)
    hamiltonian = assemble_reduced_hamiltonian(basis)
    energies, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
    wrong = eigenvectors.copy()
    wrong[:, [0, 1]] = wrong[:, [1, 0]]

    with pytest.raises(ValueError, match="residual"):
        compute_observables(
            basis,
            hamiltonian,
            energies,
            wrong,
            chunk_size=4,
        )


@pytest.mark.parametrize("state", [-1, 2**64, True, np.bool_(False)])
def test_state_inputs_are_validated_before_uint64_conversion(state):
    basis = build_orbit_basis(10)
    with pytest.raises((TypeError, ValueError), match="state"):
        project_product_state(basis, state)
    with pytest.raises((TypeError, ValueError), match="state"):
        stream_fsa_shells(basis, state, max_shell=0)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, np.bool_(False)])
def test_compute_observables_rejects_invalid_chunk_size(chunk_size):
    basis = build_orbit_basis(10)
    hamiltonian = assemble_reduced_hamiltonian(basis)
    energies, eigenvectors = np.linalg.eigh(hamiltonian.toarray())
    with pytest.raises((TypeError, ValueError), match="chunk_size"):
        compute_observables(
            basis,
            hamiltonian,
            energies,
            eigenvectors,
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    ("energies", "eigenvectors", "matrix_shape", "message"),
    [
        (np.arange(13.0), np.eye(14), (14, 14), "energies"),
        (np.arange(14.0), np.eye(13, 14), (14, 14), "eigenvectors"),
        (np.arange(14.0), np.eye(14), (13, 13), "Hamiltonian"),
        (np.r_[np.arange(13.0), np.nan], np.eye(14), (14, 14), "finite"),
    ],
)
def test_compute_observables_validates_dimensions_and_finiteness(
    energies, eigenvectors, matrix_shape, message
):
    basis = build_orbit_basis(10)
    matrix = sp.csr_matrix(matrix_shape, dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        compute_observables(basis, matrix, energies, eigenvectors, chunk_size=4)
