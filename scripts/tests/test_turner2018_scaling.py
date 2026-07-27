from turner2018_scaling import profile_sizes


def test_scaling_profile_records_basis_and_sparse_matrix_costs():
    rows = profile_sizes([8, 10])

    assert [row["length"] for row in rows] == [8, 10]
    assert [row["basis_dimension"] for row in rows] == [47, 123]
    assert all(row["hamiltonian_nnz"] > row["basis_dimension"] for row in rows)
    assert all(row["matvec_seconds"] >= 0 for row in rows)
    assert all(row["time_propagation_seconds"] >= 0 for row in rows)
