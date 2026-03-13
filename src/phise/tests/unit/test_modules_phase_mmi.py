import numpy as np
import astropy.units as u
from phise.modules import phase, mmi


def test_shift_array_and_scalar_consistency():
    # Verify that phase.shift behaves consistently when given a scalar
    # complex field or an array: the scalar result should match the
    # first element of the array result for identical inputs.
    λ = 600 * u.nm
    ψ_scalar = 1.0 + 0j
    ψ_array = np.array([1.0 + 0j, 0.0 + 0j])

    δφ = λ / 4
    out_s = phase.shift(ψ_scalar, δφ=δφ, λ=λ)
    out_a = phase.shift(ψ_array, δφ=δφ, λ=λ)

    # scalar case is equivalent to first element of array case
    assert np.allclose(out_s, out_a[0])


def test_bound_and_bound_jit_behavior():
    # Test wrapping of phase values into the canonical [0, λ[ interval.
    # We check that a scalar > λ is wrapped correctly and units are
    # preserved.
    λ = 1.0 * u.m
    # values > λ wrap around
    val = 3.25 * u.m
    wrapped = phase.bound(val, λ)
    assert np.isclose(wrapped.value, 0.25)
    assert wrapped.unit == val.unit


def test_mmi_unitary_and_nuller():
    # Validate MMI building blocks:
    # - cross_recombiner_2x2 should preserve total energy (unitary)
    # - nuller_2x2 should split a single bright input evenly between
    #   the two outputs for the simple [1,0] input.
    beams = np.array([1 + 0j, 1 + 0j])
    out_cross = mmi.cross_recombiner_2x2(beams)

    # cross recombiner is unitary -> energy preserved
    assert np.isclose(np.sum(np.abs(out_cross) ** 2), np.sum(np.abs(beams) ** 2))

    # nuller applied to [1,0] produces equal outputs
    out_n = mmi.nuller_2x2(np.array([1 + 0j, 0 + 0j]))
    assert np.allclose(out_n[0], out_n[1])
