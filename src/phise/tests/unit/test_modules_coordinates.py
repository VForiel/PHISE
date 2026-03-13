import numpy as np
import astropy.units as u
from phise.modules import coordinates


def test_get_maps_returns_quantities_and_shapes():
    N = 7
    fov = 20 * u.mas
    x, y, theta, rho = coordinates.get_maps(N=N, fov=fov)

    assert x.shape == (N, N)
    assert y.shape == (N, N)
    assert theta.shape == (N, N)
    assert rho.shape == (N, N)

    # theta and rho have astropy units
    assert hasattr(theta, 'unit')
    assert hasattr(rho, 'unit')


def test_ρθ_to_xy_edge_cases():
    fov = 10 * u.mas

    # zero separation -> origin
    # This test checks conversion from polar coordinates (ρ, θ) to
    # normalized Cartesian coordinates (u, v). It verifies two edge
    # cases: zero separation (should map to origin) and maximum
    # separation equal to fov/2 at angle π (should map to -1 on x).
    ux, uy = coordinates.ρθ_to_xy(ρ=0 * u.mas, θ=5 * u.rad, fov=fov)
    # function may return plain floats or Quantities depending on implementation
    ux_val = getattr(ux, 'value', ux)
    uy_val = getattr(uy, 'value', uy)
    assert np.allclose(ux_val, 0.0)
    assert np.allclose(uy_val, 0.0)

    # maximum radial maps: ρ = fov/2 at θ=pi -> x=-1
    ux2, uy2 = coordinates.ρθ_to_xy(ρ=fov / 2, θ=np.pi * u.rad, fov=fov)
    ux2_val = getattr(ux2, 'value', ux2)
    uy2_val = getattr(uy2, 'value', uy2)
    assert np.isclose(ux2_val, -1.0, atol=1e-9)
    assert np.isclose(uy2_val, 0.0, atol=1e-9)
