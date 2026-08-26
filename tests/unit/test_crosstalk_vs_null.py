import numpy as np

from analysis.crosstalk_vs_null import (
    random_energy_conserving_matrix,
    validate_energy_conservation,
)


def test_random_crosstalk_matrix_is_unitary_and_reaches_requested_maximum():
    rng = np.random.default_rng(42)
    requested = 0.03
    matrix = random_energy_conserving_matrix(requested, rng)
    off_diagonal = matrix - np.diag(np.diag(matrix))

    assert validate_energy_conservation(matrix)
    assert np.isclose(np.max(np.abs(off_diagonal)), requested, rtol=1e-8, atol=1e-12)
    assert np.all(np.isfinite(matrix))
