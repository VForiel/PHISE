import numpy as np
import astropy.units as u
from phise.modules import signals, test_statistics


def test_as_str_format_and_content():
    # Ensure string formatting for signal vectors lists telescope
    # indices, complex amplitude and intensity information.
    s = np.array([1 + 1j, 0.5 + 0.0j])
    out = signals.as_str(s)
    assert 'Telescope 0' in out
    assert 'Telescope 1' in out
    # intensities present
    assert '->' in out


def test_photon_flux_scaling_with_area_and_wavelength():
    # Verify photon_flux behaviour for sensible scaling:
    # - increasing collecting area increases detection rate roughly
    #   proportionally
    # - increasing wavelength reduces photon rate because each
    #   photon carries more energy (hc/λ)
    λ1 = 500 * u.nm
    λ2 = 1000 * u.nm
    Δλ = 1 * u.nm
    f = 1 * u.W / u.m**2 / u.Hz
    a1 = 1 * u.m**2
    a2 = 2 * u.m**2

    flux1 = signals.photon_flux(λ=λ1, Δλ=Δλ, f=f, a=a1, η=1.0, m=0 * u.dimensionless_unscaled)
    flux2 = signals.photon_flux(λ=λ1, Δλ=Δλ, f=f, a=a2, η=1.0, m=0 * u.dimensionless_unscaled)
    # doubling area roughly doubles flux
    assert np.isclose(flux2.value / flux1.value, 2.0, rtol=1e-6)

    # longer wavelength -> fewer photons (energy per photon ~ hc/λ)
    flux_long = signals.photon_flux(λ=λ2, Δλ=Δλ, f=f, a=a1, η=1.0, m=0 * u.dimensionless_unscaled)
    assert flux_long.value < flux1.value


def test_test_statistics_basic_helpers():
    # Basic checks for a few simple statistics wrappers: ensure they
    # compute absolute mean/median and return floats for distribution
    # comparison metrics.
    u = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 0.5, 0.5])
    assert test_statistics.mean(u, v) == abs(np.mean(u))
    assert test_statistics.median(u, v) == abs(np.median(u))
    ks = test_statistics.kolmogorov_smirnov(u, v)
    assert isinstance(ks, float)
